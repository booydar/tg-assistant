import os
import re
import json
import telebot
# from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup
from transcribe import ogg2wav, transcribe_audio
from neural import Model


class PersonaBot(telebot.TeleBot):
    def __init__(self, cred_path='creds.txt'):
        with open('config.json', 'r') as f:
            d = json.load(f)
            api_token = d['token']
            self.db_path = d['db_path']
            self.correct_chat_id = d['chat_id']
        super().__init__(api_token)
        self.tags = []
        self.model = Model()
        self.wait_value = False

        self.context = ''
        self.template = 'Ученик: {}. Кант:'

    
    def answer(self, text):

        context = self.context + self.template.format(text)
        response = self.model.generate(context)
        
        answer = self.process_response(response)
        self.context += answer

        return answer

    
    def process_response(self, response):
        ##!!!!!!!! wrong !!!!!!
        answer = response[len(context):]

        split = re.split('(\.|\?|\!)', answer)
        if '\n' in split[0]:
            answer = split[0].split('\n')[0] +  '.'
        else:
            answer = split[0]
            if len(split) > 1:
                answer += split[1]
        
        return answer


    def reset(self):
        self.context = ''
        self.Model = Model()


    def transcribe_message(self, message):
        self.tags = []
        file_info = self.get_file(message.voice.file_id)
        voice_file = self.download_file(file_info.file_path)
        with open('tmp.ogg', 'wb') as new_file:
            new_file.write(voice_file)

        wav_path = ogg2wav('tmp.ogg')
        transcription = transcribe_audio(wav_path, self.lang)
        os.system('rm tmp.*')
        return transcription

    
bot = PersonaBot()

@bot.message_handler(commands=['start'])
def start_message(message):    
    bot.chat_id = message.chat.id
    bot.send_message(message.chat.id,'Привет!\nЯ Иммануил Кант, немецкий философ.')


@bot.message_handler(content_types=['voice'])
def handle_voice(message):
    bot.chat_id = message.chat.id
    transcription = bot.transcribe_message(message)
    bot.transctiption = transcription
    bot.send_message(message.chat.id, transcription)
    answer = bot.answer(transcription)
    bot.send_message(message.chat.id, answer)


@bot.message_handler(content_types=['text'])
def handle_text(message):
    bot.chat_id = message.chat.id
    if message.text.startswith('/set'):
        bot.wait_value = message.split('/set')[1]
        bot.send_message(message.chat.id, 'set {bot.wait_value} to what value?')
    elif bot.wait_value:
        bot.model.config['generate_config'][bot.wait_value] = float(message.text)
        bot.wait_value = False
    elif message.text.startswith('/reset'):
        bot.reset()
    else:
        answer = bot.answer(message.text)
        bot.send_message(message.chat.id, answer)
    

bot.infinity_polling()