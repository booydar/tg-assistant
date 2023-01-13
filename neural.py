from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

CONFIG = {
        'model_name': "sberbank-ai/rugpt3large_based_on_gpt2",
        'model_cpt': "/home/booydar/Desktop/projects/tg_notebot/models/kantgpt_large_50ep.pth",
        # 'model_name' : 'sberbank-ai/mGPT',
        # 'model_cpt': '/home/booydar/Desktop/projects/tg_notebot/models/kant_mgpt_50ep.pth',
        'model_cls': GPT2LMHeadModel,
        'device': 'cpu',
        'generate_config' : {'do_sample':True,
                            'num_beams':1,
                            'temperature':.9,
                            'top_p':0.9,
                            'min_length': 5,
                            'max_new_tokens':100,}
        }
        
class Model:
    def __init__(self, config=CONFIG):
        model_name = config['model_name']
        self.model = config['model_cls'].from_pretrained(model_name)
        self.model.eval()

        cpt = torch.load(config['model_cpt'], map_location=config['device'])
        self.model.load_state_dict(cpt['model_state_dict'])

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model.to(config['device'])
        self.config = config


    def generate(self, text):
        max_input_size = 2048
        if 'max_length' in self.config['generate_config']:
            max_input_size -= self.config['generate_config']['max_length']
        input_ids = self.tokenizer.encode(text, return_tensors='pt').to(self.config['device'])
        if len(input_ids) > max_input_size:
            input_ids = input_ids[-max_input_size:]
        
        with torch.no_grad():
            out = self.model.generate(input_ids, **self.config['generate_config'])
        
        # print('out', out.shape)
        out = out[:, input_ids.shape[1]:]
        # print('out', out.shape)
        generated_text = list(map(self.tokenizer.decode, out))[0]
        return generated_text