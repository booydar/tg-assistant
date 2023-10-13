import os
import re
import gc
import json
import telebot
import torch
from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer
from transcribe import transcribe_audio, Punctuator

# torch.cuda.set_per_process_memory_fraction(0.9, "cuda:0")
MAX_CONTEXT_TOKENS = 800

GEN_CONFIG = {'do_sample': True,
                'num_beams':1,
                'temperature':0.7,
                'top_p':0.95,
                'repetition_penalty':1.15,
                'max_new_tokens':256,}

class PersonaBot(telebot.TeleBot):
    def __init__(self, api_token, model_name):       
        super().__init__(api_token)
        self.wait_value = False
        self.l_prompt_text = "<|system|>You are a helpful assistant. Continue the dialogue with user.</s><|prompter|>"
        self.r_prompt_text = "</s><|assistant|>"
        self.context = ''
        self.generate_config = GEN_CONFIG
        self.init_model(model_name)
        self.lang = 'en-US'
        self.model_name = model_name

    def init_model(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                    unk_token="<unk>",
                                                    bos_token="<s>",
                                                    eos_token="</s>")
        self.model = AutoGPTQForCausalLM.from_quantized(model_name,
                    disable_exllama=True,
                    use_safetensors=True,
                    trust_remote_code=False,
                    device="cuda:0",
                    use_triton=False,
                    quantize_config=None)
        self.model.eval()
        self.left_prompt = self.tokenizer.encode(self.l_prompt_text, return_tensors='pt', add_special_tokens=False)
        self.right_prompt = self.tokenizer.encode(self.r_prompt_text, return_tensors='pt', add_special_tokens=False)

    def answer_message(self, message=None, context=''):
        full_context = context
        if message is not None:
            full_context += f"\nUser: {message}"
        # print(full_context, message, context)

        input_ids = self.tokenizer.encode(full_context, return_tensors='pt', add_special_tokens=False)
        
        if input_ids.shape[1] > MAX_CONTEXT_TOKENS:
            print(f'Input length exceeded {MAX_CONTEXT_TOKENS} tokens. Truncating.')
        input_ids = input_ids[:, -MAX_CONTEXT_TOKENS:]
        input_ids = torch.cat([self.left_prompt, input_ids, self.right_prompt], dim=1).cuda()

        with torch.no_grad():
            out = self.model.generate(inputs=input_ids, **self.generate_config).cpu()
        gc.collect()
        torch.cuda.empty_cache()

        generated_text = self.tokenizer.decode(out[0], add_special_tokens=False)
        if generated_text[-4:] == '</s>':
            generated_text = generated_text[:-4]
        answer = generated_text[generated_text.index('<|assistant|>') + len('<|assistant|>'):]

        full_context += f"\nAssistant: {answer}"
        return answer, full_context

    def transcribe_message(self, message):
        self.tags = []
        file_info = self.get_file(message.voice.file_id)
        voice_file = self.download_file(file_info.file_path)
        with open("tmp.ogg", "wb") as new_file:
            new_file.write(voice_file)

        raw = transcribe_audio("tmp.ogg", self.lang)
        os.system("rm tmp.ogg")
        punctuated = punct.apply(raw)
        return punctuated

    def reset(self):
        self.context = ''
        gc.collect()
        torch.cuda.empty_cache()

    def get_context(self):
        return f"Context: \n{self.context}"

with open('config.json', 'r') as f:
    config = json.load(f)
bot = PersonaBot(config['tg_api_token'], model_name=config['model_name'])
punct = Punctuator(MODEL_PATH="../models/silero/v2_4lang_q.pt")

def continue_markup():
    markup = InlineKeyboardMarkup()
    markup.row_width = 1
    markup.add(InlineKeyboardButton("Continue", callback_data="continue"))
    return markup

@bot.callback_query_handler(func=lambda call: True)
def callback_query(call):
    if call.data == "continue":
        generated, bot.context = bot.answer_message(context=bot.context)
        bot.send_message(bot.chat_id, generated, reply_markup=continue_markup())

@bot.message_handler(commands=['start'])
def start_message(message):    
    bot.chat_id = message.chat.id
    bot.send_message(message.chat.id, bot.start_message)


@bot.message_handler(content_types=['voice'])
def handle_voice(message):
    bot.chat_id = message.chat.id
    transcription = bot.transcribe_message(message)
    bot.transctiption = transcription
    bot.send_message(message.chat.id, f'"{transcription}"')
    try:
        answer, bot.context = bot.answer_message(message=transcription, context=bot.context)
        bot.send_message(message.chat.id, answer, reply_markup=continue_markup())
    except Exception as e:
        bot.send_message(message.chat.id, f'Exception:\n{e}')


@bot.message_handler(content_types=['text'])
def handle_text(message):
    bot.chat_id = message.chat.id
    if message.text.startswith('/set_'):
        bot.wait_value = message.text.split('/set_')[1]
        bot.send_message(message.chat.id, f'set {bot.wait_value} to what value?')
    elif bot.wait_value:
        if '.' in message.text:
            bot.generate_config[bot.wait_value] = float(message.text)
        else:
            bot.generate_config[bot.wait_value] = int(message.text)
        bot.wait_value = False
    elif message.text.startswith('/reset'):
        bot.send_message(message.chat.id, "Memory erased.")
        bot.reset()
    elif message.text.startswith('/context'):
        bot.send_message(message.chat.id, bot.get_context())
        bot.reset()
    elif message.text.startswith('/config'):
        msg = '; '.join([f'{k}-{v}' for k, v in bot.generate_config.items()])
        bot.send_message(message.chat.id, msg)
    else:        
        try:
            answer, bot.context = bot.answer_message(message=message.text, context=bot.context)
            bot.send_message(message.chat.id, answer, reply_markup=continue_markup())
        except Exception as e:
            bot.send_message(message.chat.id, f'Exception:\n{e}')
    

bot.infinity_polling()