import socket
import client_protocol

IP = "127.0.0.1"
PORT = 12345
genres = [
            "<superhero>",
            "<action>",
            "<drama>",
            "<thriller>",
            "<horror>",
            "<sci_fi>",
        ]

def main():
    client_socket = socket.socket()
    client_socket.connect((IP, PORT))
    try:
        while True:

            genre_check=False
            prompt = input("Enter your prompt:")
            while genre_check == False:
                genre = input("Enter desired genre (1.superhero 2.action 3.drama 4.thriller 5.horror 6.sci-fi)")
                genre=int(genre)
                if 0 < genre < 7:
                    genre=genres[genre-1]
                    genre_check = True
                else:
                    print("The genre you requested is not yet supported XD")
            word_count = input("Enter desired word count:")
            client_request_str = prompt + "/" + genre + "/" + word_count
            if client_request_str:
                client_protocol.send_request(client_socket, client_request_str)
                response = client_protocol.get_response(client_socket)
                if response:
                    print("Server:", response)
                else:
                    if not response:
                        print("Server crashes")
                    break
    finally:
        client_socket.close()

if __name__ == '__main__':
    main()