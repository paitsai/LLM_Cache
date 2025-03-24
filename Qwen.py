import numpy as np
from transformers import AutoTokenizer, AutoConfig
from onnxruntime import InferenceSession, SessionOptions, ExecutionMode, OrtValue
import os
import time
MODEL_PATH="/root/autodl-tmp/codes/Qwen_1.5B_onnx"

class QwenGPUOptimized:
    def __init__(self, MODEL_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.config = AutoConfig.from_pretrained(MODEL_PATH)
        
        # 模型参数
        self.NUM_ATTN_LAYER = self.config.num_hidden_layers
        self.NUM_HEAD = self.config.num_key_value_heads
        self.HEAD_DIM = self.config.hidden_size // self.config.num_attention_heads
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        
        # GPU配置
        self.providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'gpu_mem_limit': 30 * 1024 * 1024 * 1024,
            }),
            'CPUExecutionProvider'
        ]
        
        self.sess_options = SessionOptions()
        self.session = InferenceSession(
            os.path.join(MODEL_PATH, "model.onnx"),
            providers=self.providers,
            sess_options=self.sess_options
        )
        
        self.use_gpu = 'CUDAExecutionProvider' in self.session.get_providers()
        print(f"GPU加速已{'启用' if self.use_gpu else '禁用'}")

    def init_kv_cache(self):
        """初始化KV缓存"""
        cache = {}
        empty_tensor = np.zeros((1, self.NUM_HEAD, 0, self.HEAD_DIM), dtype=np.float32)
        
        for i in range(self.NUM_ATTN_LAYER):
            if self.use_gpu:
                cache[f"past_key_values.{i}.key"] = OrtValue.ortvalue_from_numpy(empty_tensor, 'cuda', 0)
                cache[f"past_key_values.{i}.value"] = OrtValue.ortvalue_from_numpy(empty_tensor, 'cuda', 0)
            else:
                cache[f"past_key_values.{i}.key"] = empty_tensor.copy()
                cache[f"past_key_values.{i}.value"] = empty_tensor.copy()
        return cache

    def _prepare_inputs(self, input_ids, past_seq_length, past_key_values):
        """准备输入数据"""
        base_inputs = {
            "input_ids": input_ids,
            "attention_mask": np.ones_like(input_ids),
            "position_ids": np.array([[past_seq_length]], dtype=np.int64)
        }
        
        if self.use_gpu:
            gpu_inputs = {}
            for k, v in base_inputs.items():
                gpu_inputs[k] = OrtValue.ortvalue_from_numpy(v, 'cuda', 0)
            return {**gpu_inputs, **past_key_values}
        return {**base_inputs, **past_key_values}

    def _update_kv_cache(self, past_key_values, new_kv, keep_last_k=None):
        """更新KV缓存并保留最近K个token的缓存
        
        Args:
            past_key_values: 历史KV缓存
            new_kv: 新生成的KV缓存
            keep_last_k: 保留的token数量(None表示保留全部)
        """
        updated_kv = {}
        
        for i in range(self.NUM_ATTN_LAYER):
            # 获取历史和新KV
            past_key = past_key_values[f"past_key_values.{i}.key"]
            past_val = past_key_values[f"past_key_values.{i}.value"]
            new_key = new_kv[f"past_key_values.{i}.key"]
            new_val = new_kv[f"past_key_values.{i}.value"]
            
            # 转换为numpy处理
            past_key_np = past_key.numpy() if hasattr(past_key, 'numpy') else past_key
            past_val_np = past_val.numpy() if hasattr(past_val, 'numpy') else past_val
            new_key_np = new_key.numpy() if hasattr(new_key, 'numpy') else new_key
            new_val_np = new_val.numpy() if hasattr(new_val, 'numpy') else new_val
            
            # 拼接新旧KV
            concat_key = np.concatenate([past_key_np, new_key_np], axis=2)
            concat_val = np.concatenate([past_val_np, new_val_np], axis=2)
            
            # 裁剪保留最近K个
            if keep_last_k is not None:
                start_idx = max(0, concat_key.shape[2] - keep_last_k)
                concat_key = concat_key[:, :, start_idx:, :]
                concat_val = concat_val[:, :, start_idx:, :]
            
            # 转换回原始格式
            if self.use_gpu:
                updated_kv[f"past_key_values.{i}.key"] = OrtValue.ortvalue_from_numpy(concat_key, 'cuda', 0)
                updated_kv[f"past_key_values.{i}.value"] = OrtValue.ortvalue_from_numpy(concat_val, 'cuda', 0)
            else:
                updated_kv[f"past_key_values.{i}.key"] = concat_key
                updated_kv[f"past_key_values.{i}.value"] = concat_val
                
        return updated_kv

        
    def generate_token(self, input_ids, past_key_values, past_seq_length):
        """生成单个token"""
        inputs = self._prepare_inputs(input_ids, past_seq_length, past_key_values)
        outputs = self.session.run(None, inputs)
        
        # 解析输出
        logits = outputs[0]
        new_kv = {}
        for i in range(self.NUM_ATTN_LAYER):
            new_kv[f"past_key_values.{i}.key"] = outputs[1 + 2*i]
            new_kv[f"past_key_values.{i}.value"] = outputs[2 + 2*i]
        
        # 更新缓存
        updated_kv = self._update_kv_cache(past_key_values, new_kv, keep_last_k=72)
        
        # 采样
        next_token_id = np.argmax(logits[:, -1, :], axis=-1)[0]
        return next_token_id, updated_kv

    def generate_dialogue(self, prompt, max_length=64):
        """生成对话"""
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        past_kv = self.init_kv_cache()
        past_seq_len = input_ids.shape[1]
        generated_ids = input_ids.tolist()[0]
        
        for _ in range(max_length):
            next_token_id, past_kv = self.generate_token(
                input_ids=np.array([[generated_ids[-1]]]),
                past_key_values=past_kv,
                past_seq_length=past_seq_len
            )
            
            generated_ids.append(next_token_id)
            past_seq_len += 1
            print(self.tokenizer.decode([next_token_id], skip_special_tokens=True), end="", flush=True)
            
            if next_token_id == self.eos_token_id:
                break
                
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


# 使用示例
if __name__ == "__main__":
    MODEL_PATH = "/root/autodl-tmp/codes/Qwen_1.5B_onnx"
    model = QwenGPUOptimized(MODEL_PATH)
    print(model.generate_dialogue("关于李白"))
