message_length = 4


def get_response(client_socket):
    len_response_str = client_socket.recv(message_length)
    len_response_str = len_response_str.decode()
    len_response = int(len_response_str)
    response = client_socket.recv(len_response)
    response = response.strip()
    response_str = response.decode()
    return response_str


def send_request(client_socket, request_str):
    request_len_str = str(len(request_str)).zfill(message_length)
    request_str = "".join([request_len_str, request_str])
    request = request_str.encode()
    client_socket.send(request)