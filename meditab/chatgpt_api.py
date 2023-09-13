import os
import pdb
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

class chatGPTAPI:
    def __init__(self, system_message="You are a helpful assistant.", keypath='./openai_key.txt', orgid=None):
        '''args:

        keypath: path to the API key file
            It should contain only one line, the API key in your OpenAI account.

        orgid: string
            The organization ID displayed in the openai account page.

        system_message: string
            The message that controls the behavior of the chatgpt model.

        Usage:
            >>> chat = chatGPTAPI()
            >>> chat.get_response_oneround("Hello")
        '''
        openai.organization = orgid
        with open(keypath, 'r') as f:
            openai.api_key = f.read().strip()

        self.system_message = system_message

        # create a new ChatCompletion
        self.messages = [
            {"role": "system", "content": system_message},   
        ]

    def get_response_oneround(self, chat):
        '''
        Build a one-round chat with the GPT-3 model.

        args:
            chat: string
        '''
        # copy self.messages (a list of dict) into a new list named message
        message = self.messages.copy()
        message.append(self._build_user_message(chat))

        # build message
        response = self._completion_with_backoff(
            model="gpt-3.5-turbo",
            messages=message,
        )
        
        return response["choices"][0]["message"]["content"]

    def reset_message(self):
        '''
        reset the messages in the chat history.
        '''
        self.messages = [
            {"role": "system", "content": self.system_message},   
        ]

    def _build_user_message(self, chat):
        return {"role": "user", "content": chat}

    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(60))
    def _completion_with_backoff(self, **kwargs):
        return openai.ChatCompletion.create(**kwargs)