message_length = 4


def get_request(client_socket):
    len_request_str = client_socket.recv(message_length)
    len_request_str = len_request_str.decode()
    len_request = int(len_request_str)
    request = client_socket.recv(len_request)
    request = request.strip()
    request_str = request.decode()
    return request_str


def send_response(client_socket, response_str):
    response_len_str = str(len(response_str)).zfill(message_length)
    response_str = "".join([response_len_str, response_str])
    response = response_str.encode()
    client_socket.send(response)
