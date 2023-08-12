import os
import re
import json
import telebot
import torch
from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup
from transformers import AutoModelForCausalLM, AutoTokenizer
from transcribe import transcribe_audio

GEN_CONFIG = {'num_beams':1,
                'temperature':1.2,
                'top_p':0.9,
                'min_length': 5,
                'max_new_tokens':150,}

class PersonaBot(telebot.TeleBot):
    def __init__(self, api_token, model_name):       
        super().__init__(api_token)
        self.wait_value = False
        self.prompt = "<|prompter|>{}<|endoftext|><|assistant|>"
        self.context = ''
        self.generate_config = GEN_CONFIG
        self.init_model(model_name)
        self.lang = 'en-US'
        self.model_name = model_name

    def init_model(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map='auto', trust_remote_code=True)#, load_in_8bit_fp32_cpu_offload=True)
        self.model.eval()

    def answer_message(self, message, context):
        if not context:
            model_input = self.prompt.format(message)
            updated_context = f"User:{message}"
        else:
            updated_context = f"{context}\nUser:{message}"
            model_input = self.prompt.format(updated_context)

        input_ids = self.tokenizer.encode(model_input, return_tensors='pt').cuda()

        with torch.no_grad():
            out = self.model.generate(input_ids, do_sample=True, **self.generate_config)

        generated_text = list(map(self.tokenizer.decode, out))[0]
        self.last_generation = generated_text
        answer = generated_text[generated_text.index('<|assistant|>') + len('<|assistant|>'):]
        if '<|endoftext|>' in answer:
            answer = answer[:answer.index('<|endoftext|>')]

        updated_context += f"\nAssistant:{answer}"
        return answer, updated_context

    def continue_message(self, context):
        model_input = self.last_generation
        if model_input[-13:] == '<|endoftext|>':
            model_input = model_input[-13:]
        input_ids = self.tokenizer.encode(model_input, return_tensors='pt').cuda()

        with torch.no_grad():
            out = self.model.generate(input_ids, **self.generate_config)

        generated_text = list(map(self.tokenizer.decode, out))[0]
        self.last_generation = generated_text
        answer = generated_text[generated_text.index('<|assistant|>') + len('<|assistant|>'):]
        if '<|endoftext|>' in answer:
            answer = answer[:answer.index('<|endoftext|>')]

        start_pos = generated_text.index('<|prompter|>') + len('<|prompter|>')
        end_pos = generated_text.index('<|endoftext|>')
        updated_context = generated_text[start_pos:end_pos] + answer
        return answer, updated_context

        # with torch.no_grad():
        #     out = self.model.generate(input_ids, **self.generate_config)

        # generated_text = list(map(self.tokenizer.decode, out))[0]
        # answer = generated_text[generated_text.index('<|assistant|>') + len('<|assistant|>'):]
        # if '<|endoftext|>' in answer:
        #     answer = answer[:answer.index('<|endoftext|>')]

        # updated_context = context + answer
        # return answer, updated_context

    def transcribe_message(self, message):
        self.tags = []
        file_info = self.get_file(message.voice.file_id)
        voice_file = self.download_file(file_info.file_path)
        with open("tmp.ogg", "wb") as new_file:
            new_file.write(voice_file)

        raw = transcribe_audio("tmp.ogg", self.lang)
        os.system("rm tmp.ogg")
        # punctuated = punct.apply(raw)
        # return raw, punctuated
        return raw

    def reset(self):
        self.context = ''
        self.init_model(self.model_name)

    def get_context(self):
        return f"Context: \n{self.context}"

with open('/home/booydar/Desktop/projects/tg_notebot/assistant_bot/config.json', 'r') as f:
    config = json.load(f)
bot = PersonaBot(config['tg_api_token'], model_name=config['model_name'])


def continue_markup():
    markup = InlineKeyboardMarkup()
    markup.row_width = 1
    markup.add(InlineKeyboardButton("Continue", callback_data="continue"))
    return markup

@bot.callback_query_handler(func=lambda call: True)
def callback_query(call):
    if call.data == "continue":
        generated, bot.context = bot.continue_message(bot.context)
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
    # bot.send_message(message.chat.id, f'You said:\n "{transcription}".\nThinking about it.')
    bot.send_message(message.chat.id, f'"{transcription}"')
    answer, bot.context = bot.answer_message(transcription, bot.context)
    bot.send_message(message.chat.id, answer, reply_markup=continue_markup())


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
        answer, bot.context = bot.answer_message(message.text, bot.context)
        bot.send_message(message.chat.id, answer, reply_markup=continue_markup())
    

bot.infinity_polling()