import numpy as np
from transformers import AutoTokenizer, AutoConfig
from onnxruntime import InferenceSession
import os

MODEL_PATH="/root/autodl-tmp/codes/Qwen_1.5B_onnx"

class Qwen:
    def __init__(self, MODEL_PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.session = InferenceSession(os.path.join(MODEL_PATH, "model.onnx"))
        self.config = AutoConfig.from_pretrained(MODEL_PATH)
        self.NUM_ATTN_LAYER = self.config.num_hidden_layers
        self.NUM_HEAD = self.config.num_key_value_heads
        self.HEAD_DIM = self.config.hidden_size // self.config.num_attention_heads
        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        
    def init_kv_cache(self):
        cache = {}
        num_layers=self.NUM_ATTN_LAYER
        num_heads=self.NUM_HEAD
        head_dim=self.HEAD_DIM
        for i in range(num_layers):
            cache[f"past_key_values.{i}.key"] = np.zeros(
                (1, num_heads, 0, head_dim), dtype=np.float32
            )
            cache[f"past_key_values.{i}.value"] = np.zeros(
                (1, num_heads, 0, head_dim), dtype=np.float32
            )
        return cache
    
    def generate_token(self, input_ids, past_key_values, past_seq_length):
        """生成单个token并更新KV Cache"""
        # 准备模型输入
        inputs = {
            "input_ids": input_ids,
            "attention_mask": np.ones_like(input_ids),  # 全1表示全部可见
            "position_ids": np.array([[past_seq_length]], dtype=np.int64),
            **past_key_values
        }
        
        # 运行推理
        outputs = self.session.run(None, inputs)
        num_layers=self.NUM_ATTN_LAYER
        # 解析输出 [logits, k0, v0, k1, v1,...]
        logits = outputs[0]  # [1, 1, vocab_size]
        new_kv = {}
        for i in range(num_layers):
            new_kv[f"past_key_values.{i}.key"] = outputs[1 + 2*i]    # 第i层的key
            new_kv[f"past_key_values.{i}.value"] = outputs[2 + 2*i]  # 第i层的value
        
        # 更新KV Cache（拼接历史）
        updated_kv = {}
        for i in range(num_layers):
            updated_kv[f"past_key_values.{i}.key"] = np.concatenate([
                past_key_values[f"past_key_values.{i}.key"],
                new_kv[f"past_key_values.{i}.key"]
            ], axis=2)  # 沿序列长度维度拼接
            
            updated_kv[f"past_key_values.{i}.value"] = np.concatenate([
                past_key_values[f"past_key_values.{i}.value"],
                new_kv[f"past_key_values.{i}.value"]
            ], axis=2)
        
        # 采样下一个token（贪婪解码）
        next_token_id = np.argmax(logits[:, -1, :], axis=-1)[0]
        
        return next_token_id, updated_kv

    def generate_dialogue(self, prompt, max_length=10):
        """生成完整对话"""
        # 编码初始输入
        input_ids = self.tokenizer.encode(prompt, return_tensors="np")
        past_kv = self.init_kv_cache()
        past_seq_len = input_ids.shape[1]  # 初始序列长度
        
        # 存储所有生成的token
        generated_ids = input_ids.tolist()[0]
        
        for _ in range(max_length):
            # 生成单个token
            next_token_id, past_kv = self.generate_token(
                input_ids=np.array([[generated_ids[-1]]]),  # 每次只传入最新token
                past_key_values=past_kv,
                past_seq_length=past_seq_len
            )
            
            # 更新状态
            generated_ids.append(next_token_id)
            past_seq_len += 1
            
            # 打印实时结果
            print(self.tokenizer.decode([next_token_id], skip_special_tokens=True), end="", flush=True)
            
            # 遇到终止符则停止
            if next_token_id == self.eos_token_id:
                break
        
        # 返回完整文本
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    # 使用示例
    

            
        