import socket
import torch
import threading
import server_protocol
from transformers import TextGenerationPipeline, GPT2LMHeadModel, AutoTokenizer


IP = "0.0.0.0"
PORT = 12345
checkpoint = "story_generator_checkpoint"
model = GPT2LMHeadModel.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
story_generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)


def get_pred(text, genre, word_count):
    input_prompt = "<BOS> " + genre + " " + text
    story = story_generator(input_prompt, max_length=word_count, do_sample=True,
                            repetition_penalty=1.1, temperature=1.2,
                            top_p=0.95, top_k=50)
    story = story[0]
    story = story['generated_text']
    return story


def create_and_send_response(request_str, client_socket):
    request = request_str.split("/")
    prompt, genre, word_count = request
    print(word_count)
    word_count = int(word_count)
    prompt = get_pred(prompt, genre, word_count)
    response_str = prompt
    server_protocol.send_response(client_socket, response_str)

    return request_str and response_str != "Bye!"


def conversation(client_socket, client_address):
    ip, port = client_address
    print("Connection made. Client ip  =", ip, ", Client port =", port)

    ack = True
    try:
        while ack:

            request = server_protocol.get_request(client_socket)
            print("Client:", request)
            ack = create_and_send_response(request, client_socket)
    finally:
        client_socket.close()
    print("Connection severed. Client ip  =", ip, ", Client port =", port)


def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((IP, PORT))
    server_socket.listen(1)
    print("Server is up and running.")

    try:
        while True:
            client_socket, client_address = server_socket.accept()
            thread_for_client = threading.Thread(target=conversation, args=(client_socket, client_address))
            thread_for_client.start()
    finally:
        server_socket.close()


if __name__ == '__main__':
    main()