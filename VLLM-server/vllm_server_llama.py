import datetime
import time


if __name__=='__main__':
    import os 
    from modeling_spsr_llama import SPSRLlamaForCausalLM
    from vllm import ModelRegistry
    ModelRegistry.register_model("SPSRLlamaForCausalLM", SPSRLlamaForCausalLM)

    from vllm import AsyncEngineArgs,AsyncLLMEngine
    from vllm.sampling_params import SamplingParams
    # from modelscope import AutoTokenizer, GenerationConfig,snapshot_download
    from transformers import GenerationConfig,AutoTokenizer
    from huggingface_hub import snapshot_download
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse
    import uvicorn
    from prompt_utils import _build_prompt,remove_stop_words
    import uuid
    import json 

    # http接口服务
    app=FastAPI()

    # vLLM参数
    base_model_path = "/home/jim/nas/yzg/Llama-3-8b/base"  # 基础模型路径
    model_dir = "/media/ssd/yzg/SPSR-GPTQ-Compressor/checkpoints/Llama3-8B-spsr-8"
    tensor_parallel_size=1
    gpu_memory_utilization=0.8
    dtype='float16'

    MASK_64_BITS = (1 << 64) - 1
    def random_uuid() -> str:
        return f"{uuid.uuid4().int & MASK_64_BITS:016x}"  # 16 hex char

    # vLLM模型加载
    def load_vllm():
        global generation_config,tokenizer,stop_words_ids,engine

        # 1. 先加载原始（或已 SPSR 保存的）transformers 配置与分词器
        # model_dir: 用于 vLLM 目录，依赖于 Base Model 使用的 config

        generation_config = GenerationConfig.from_pretrained(base_model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, fast_tokenizer=True)
        tokenizer.eos_token_id = generation_config.eos_token_id
        stop_words_ids = [128009]

        # 2. vLLM 参数配置：指定自定义模型名和 SPSR 元信息
        args = AsyncEngineArgs(model_dir)
        args.worker_use_ray = False
        args.engine_use_ray = False
        args.tokenizer = model_dir
        args.tensor_parallel_size = tensor_parallel_size
        args.trust_remote_code = True
        args.gpu_memory_utilization = gpu_memory_utilization
        args.dtype = dtype
        args.max_num_seqs = 20

        # 3. 加载引擎
        engine = AsyncLLMEngine.from_engine_args(args)

        return generation_config, tokenizer, stop_words_ids, engine

    generation_config,tokenizer,stop_words_ids,engine=load_vllm()

    # 用户停止句匹配
    def match_user_stop_words(response_token_ids,user_stop_tokens):
        for stop_tokens in user_stop_tokens:
            if len(response_token_ids)<len(stop_tokens):
                continue 
            if response_token_ids[-len(stop_tokens):]==stop_tokens:
                return True  # 命中停止句, 返回True
        return False

    # chat对话接口
    @app.post("/chat")
    async def chat(request: Request):
        request=await request.json()
        
        query=request.get('query',None)
        history=request.get('history',[])
        system=request.get('system','You are a helpful assistant.')
        stream=request.get("stream",False)
        user_stop_words=request.get("user_stop_words",[])    # list[str]，用户自定义停止句，例如：['Observation: ', 'Action: ']定义了2个停止句，遇到任何一个都会停止
        
        if query is None:
            return Response(status_code=502,content='query is empty')

        # 用户停止词
        user_stop_tokens=[]
        for words in user_stop_words:
            user_stop_tokens.append(tokenizer.encode(words))
        
        # 构造prompt
        # prompt_text,prompt_tokens=_build_prompt(generation_config,tokenizer,query,history=history,system=system)
        prompt_text = f"Human:{query}"
        # vLLM请求配置
        sampling_params=SamplingParams(
                                        top_p=generation_config.top_p,
                                        top_k=-1 if generation_config.top_k == 0 else generation_config.top_k,
                                        temperature=generation_config.temperature,
                                        repetition_penalty=generation_config.repetition_penalty,
                                        max_tokens=generation_config.max_new_tokens)
        # vLLM异步推理（在独立线程中阻塞执行推理，主线程异步等待完成通知）
        request_id=str(uuid.uuid4().hex)
        results_iter=engine.generate(prompt=prompt_text,sampling_params=sampling_params,request_id=request_id)
        
        # 流式返回，即迭代transformer的每一步推理结果并反复返回
        if stream:
            async def streaming_resp():
                async for result in results_iter:
                    # 移除im_end,eos等系统停止词
                    token_ids=remove_stop_words(result.outputs[0].token_ids,stop_words_ids)
                    # 返回截止目前的tokens输出                
                    text=tokenizer.decode(token_ids)
                    yield (json.dumps({'text':text})+'\0').encode('utf-8')
                    # 匹配用户停止词,终止推理
                    if match_user_stop_words(token_ids,user_stop_tokens):
                        await engine.abort(request_id)   # 终止vllm后续推理
                        break
            return StreamingResponse(streaming_resp())

        # 整体一次性返回模式
        async for result in results_iter:
            # 移除im_end,eos等系统停止词
            token_ids=remove_stop_words(result.outputs[0].token_ids,stop_words_ids)
            # 返回截止目前的tokens输出                
            text=tokenizer.decode(token_ids)
            # 匹配用户停止词,终止推理
            if match_user_stop_words(token_ids,user_stop_tokens):
                await engine.abort(request_id)   # 终止vllm后续推理
                break

        ret={"text":text}
        return JSONResponse(ret)
    
    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": model_dir,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "owner",
                    "root": model_dir,
                    "parent": None,
                }
            ],
        }


    @app.post("/v1/completions")
    async def openai_completions(request: Request):
        try:
            request_data = await request.json()
            print("request_data:", request_data)

            prompt = request_data.get("prompt", "")
            if isinstance(prompt, list):
                prompt = prompt[0] if prompt else ""

            stream = request_data.get("stream", False)
            stream_options = request_data.get("stream_options", {}) or {}
            include_usage = bool(stream_options.get("include_usage", False))

            max_tokens = request_data.get(
                "max_tokens",
                request_data.get("max_completion_tokens", generation_config.max_new_tokens),
            )
            temperature = request_data.get("temperature", generation_config.temperature)
            top_p = request_data.get("top_p", generation_config.top_p)
            top_k = request_data.get(
                "top_k",
                -1 if getattr(generation_config, "top_k", 0) == 0 else generation_config.top_k,
            )
            min_p = request_data.get("min_p", 0.0)
            frequency_penalty = request_data.get("frequency_penalty", 0.0)
            presence_penalty = request_data.get("presence_penalty", 0.0)
            repetition_penalty = request_data.get("repetition_penalty", 1.0)
            ignore_eos = request_data.get("ignore_eos", False)
            stop = request_data.get("stop", None)

            stop_strings = None
            if stop:
                if isinstance(stop, str):
                    stop_strings = [stop]
                elif isinstance(stop, list):
                    stop_strings = [s for s in stop if isinstance(s, str) and s]

            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                repetition_penalty=repetition_penalty,
                stop=stop_strings,
                ignore_eos=ignore_eos,
            )

            # benchmark 会把 request_id 放在 header: x-request-id
            request_id = request.headers.get("x-request-id") or request_data.get(
                "request_id", f"cmpl-{random_uuid()}"
            )
            created_time = int(time.time())

            results_iter = engine.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            )

            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))

            if stream:
                async def event_stream():
                    prev_text = ""
                    last_finish_reason = None
                    final_output = None

                    async for result in results_iter:
                        final_output = result
                        output = result.outputs[0]
                        full_text = output.text or ""
                        last_finish_reason = getattr(output, "finish_reason", None)

                        if full_text.startswith(prev_text):
                            delta_text = full_text[len(prev_text):]
                        else:
                            delta_text = full_text

                        prev_text = full_text

                        # 即使 delta_text 为空，也保持兼容 OpenAI / benchmark 行为
                        chunk = {
                            "id": request_id,
                            "object": "text_completion",
                            "created": created_time,
                            "model": model_dir,
                            "choices": [
                                {
                                    "text": delta_text,
                                    "index": 0,
                                    "finish_reason": None,
                                }
                            ],
                        }
                        print("stream chunk:", chunk)
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                    # 最终 completion chunk
                    final_chunk = {
                        "id": request_id,
                        "object": "text_completion",
                        "created": created_time,
                        "model": model_dir,
                        "choices": [
                            {
                                "text": "",
                                "index": 0,
                                "finish_reason": last_finish_reason or "stop",
                            }
                        ],
                    }
                    print("final chunk:", final_chunk)
                    yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"

                    # 关键：benchmark 需要 usage chunk
                    if include_usage and final_output is not None and final_output.outputs:
                        output = final_output.outputs[0]
                        completion_tokens = len(output.token_ids or [])

                        usage_chunk = {
                            "id": request_id,
                            "object": "text_completion",
                            "created": created_time,
                            "model": model_dir,
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": prompt_tokens + completion_tokens,
                            },
                        }
                        print("usage chunk:", usage_chunk)
                        yield f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n"

                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    event_stream(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    },
                )

            # 非流式
            final_result = None
            async for result in results_iter:
                final_result = result

            if final_result is None or not final_result.outputs:
                return JSONResponse(
                    {
                        "error": {
                            "message": "No completion generated",
                            "type": "server_error",
                            "code": 500,
                        }
                    },
                    status_code=500,
                )

            output = final_result.outputs[0]
            generated_text = output.text or ""
            completion_tokens = len(output.token_ids or [])
            finish_reason = getattr(output, "finish_reason", None) or "stop"

            response = {
                "id": request_id,
                "object": "text_completion",
                "created": created_time,
                "model": model_dir,
                "choices": [
                    {
                        "text": generated_text,
                        "index": 0,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }

            print("non-stream response:", response)
            return JSONResponse(response)

        except Exception as e:
            import traceback
            print(f"Error in /v1/completions: {e}\n{traceback.format_exc()}")
            return JSONResponse(
                {
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "code": 500,
                    }
                },
                status_code=500,
            )
        
    if __name__=='__main__':
        uvicorn.run(app,
                    host=None,
                    port=8000,
                    log_level="debug")