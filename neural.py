from transformers import GPT2LMHeadModel, GPTNeoForCausalLM, GPT2Tokenizer
import torch

CONFIG = {
        'model_name': "sberbank-ai/rugpt3medium_based_on_gpt2",
        'model_cpt': "~/Desktop/projects/tg_notebot/models/kantgpt_medium_20ep.pth",
        'model_cls': GPT2LMHeadModel,
        'device': 'cuda',
        'generate_config' : {'do_sample':True,
                            'num_beams':3,
                            'temperature':.9,
                            'top_p':1.3,
                            'min_length': 5,
                            'max_length':100,}
        }
        
class Model:
    def __init__(self, config=CONFIG):
        model_name = config['model_name']
        self.model = config['model_cls'].from_pretrained(model_name)
        self.model.eval()

        cpt = torch.load(config['model_cpt'])
        self.model.load_state_dict(cpt['model_state_dict'])

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model.to(config['device'])
        self.config = config


    def generate(self, text):
        max_input_size = 2048 - self.config['generate_config']['max_length']
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.config['device'])
        if len(input_ids) > max_input_size:
            input_ids = input_ids[-max_input_size:]
        
        with torch.no_grad():
            out = self.model.generate(input_ids, **self.config['generate_config'])
        
        generated_text = list(map(self.tokenizer.decode, out))[0]
        return generated_text