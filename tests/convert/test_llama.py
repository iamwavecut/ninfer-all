"""Llama import keeps encoded rows exactly, reversing GGUF's rotary-pair permutation."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from tools.convert import llama
from tools.convert.sources.gguf import GGUFFile, write_gguf, TYPE_F32
from tools.convert.pipeline import convert
from tools.convert.recipe import Recipe
from tools.artifact.reader import Artifact


def config(**change):
    out=dict(architectures=['LlamaForCausalLM'],model_type='llama',hidden_act='silu',
             hidden_size=256,vocab_size=256,num_hidden_layers=1,max_position_embeddings=262144,
             num_attention_heads=2,num_key_value_heads=1,head_dim=128,intermediate_size=256,
             rope_theta=120000000.,rms_norm_eps=1e-5,tie_word_embeddings=False)
    return out|change


@pytest.mark.parametrize('change',[{'hidden_act':'gelu'},{'attention_bias':True},
    {'mlp_bias':True},{'tie_word_embeddings':True},{'rope_scaling':{'type':'linear','factor':2}},
    {'head_dim':64},{'num_key_value_heads':3}])
def test_reject_different_mathematics(change):
    with pytest.raises(ValueError):llama.text_config(config(**change))


def test_encoded_rows_and_rotary_permutation(tmp_path):
    hf=tmp_path/'hf';hf.mkdir()
    (hf/'tokenizer.json').write_text(json.dumps({'model':{'type':'BPE','vocab':{f't{i}':i for i in range(256)},'merges':[]}}))
    (hf/'tokenizer_config.json').write_text('{}')
    (hf/'chat_template.jinja').write_text('{{ messages }}')
    (hf/'generation_config.json').write_text('{}')
    gen=np.random.default_rng(731);stored={};tensors=[]
    def matrix(name,n,k):
        b=gen.integers(0,256,(n,k//32,34),dtype=np.uint8)
        b[:,:,:2]=np.array([.0125],dtype=np.float16).view(np.uint8)
        stored[name]=b.reshape(n,-1).copy();tensors.append((name,(n,k),8,b.tobytes()))
    def vector(name):
        v=gen.normal(size=256).astype(np.float32);stored[name]=v;tensors.append((name,(256,),TYPE_F32,v.tobytes()))
    matrix('token_embd.weight',256,256);matrix('output.weight',256,256);vector('output_norm.weight')
    for name,n,k in [('attn_q',256,256),('attn_k',128,256),('attn_v',128,256),('attn_output',256,256),
                     ('ffn_gate',256,256),('ffn_up',256,256),('ffn_down',256,256)]:matrix(f'blk.0.{name}.weight',n,k)
    vector('blk.0.attn_norm.weight');vector('blk.0.ffn_norm.weight')
    kv={'general.architecture':'llama','llama.block_count':1,'llama.embedding_length':256,
        'llama.feed_forward_length':256,'llama.attention.head_count':2,'llama.attention.head_count_kv':1,
        'llama.rope.dimension_count':128,'llama.rope.freq_base':120000000.,'llama.attention.layer_norm_rms_epsilon':1e-5}
    gguf=tmp_path/'source.gguf';write_gguf(gguf,kv,tensors)
    out=tmp_path/'model.ninfer'
    with GGUFFile(gguf) as source:
        model=llama.build_model(SimpleNamespace(config=config(),root=hf),source)
        recipe=Recipe(model);llama.llama_gguf(model,recipe,{'gguf':source})
        convert(model,recipe,out,device='cpu',rows_per_chunk=31)
    with Artifact(out) as artifact:
        assert artifact.directory.components['text']['config']['qk_layout']=='neox'
        def read(name):return artifact.read_object(artifact.directory.bindings[name]['object'])
        for role,original,heads in [('query','attn_q',2),('key','attn_k',1)]:
            rows=stored[f'blk.0.{original}.weight'];width=rows.shape[1]
            # Independent reshape/transpose oracle, not the converter's row index mapping.
            expected=rows.reshape(heads,64,2,width).transpose(0,2,1,3).copy().reshape(heads*128,width)
            assert read(f'text/layers/0/attention/{role}')==expected.tobytes()
        assert read('text/layers/0/attention/value')==stored['blk.0.attn_v.weight'].tobytes()
        assert read('text/layers/0/mlp/down')==stored['blk.0.ffn_down.weight'].tobytes()
        assert read('text/final_norm')==stored['output_norm.weight'].tobytes()
        assert not any('query_norm' in name or 'sinks' in name for name in artifact.directory.bindings)
