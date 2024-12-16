import socket

def start_server(host, port):
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Bind the socket to the address and port
    server_socket.bind((host, port))
    
    # Listen for incoming connections
    server_socket.listen(5)
    print(f"Server listening on {host}:{port}")

    try:
        while True:
            # Accept a connection
            client_socket, client_address = server_socket.accept()
            print(f"Connection from {client_address}")
            
            try:
                # Receive data from the client
                data = client_socket.recv(1024).decode('utf-8')
                if data:
                    print(f"Received message: {data}")

                    # Send a response back to the client
                    response = "Message received"
                    client_socket.sendall(response.encode('utf-8'))
                else:
                    print("No data received")
            finally:
                # Close the client connection
                client_socket.close()
    except KeyboardInterrupt:
        print("Server shutting down...")
    finally:
        # Close the server socket
        server_socket.close()

if __name__ == "__main__":
    server_host = '0.0.0.0'  # Listen on all available interfaces
    server_port = 8080      # Port to listen on
    
    start_server(server_host, server_port)

