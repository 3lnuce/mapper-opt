import socket
import struct
import numpy as np
import time

# Define the IP address and port to communicate with the client
HOST = '0.0.0.0'  # Listen on all available interfaces
PORT = 8080       # Port to listen on
CLIENT_IP = '192.168.10.2'
CHUNK_SIZE = 4096

# Create a socket object
server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_socket.bind((HOST, PORT))

print(f"Server listening on {HOST}:{PORT}")

# Example tensor data using NumPy for demonstration
tensors = [
    np.random.randn(1000, 5).astype(np.float32),  # Tensor with 10 rows and 5 columns
    np.random.randn(1000, 3).astype(np.float32),  # Tensor with 10 rows and 3 columns
    np.random.randn(1000, 7).astype(np.float32)   # Tensor with 10 rows and 7 columns
]

print ("Tensor 1: \n", tensors[0])
print ()
print ("Tensor 2: \n", tensors[1])
print ()
print ("Tensor 3: \n", tensors[2])
print ()

def serialize_tensor(tensor):
    # Convert tensor to bytes
    rows, cols = tensor.shape
    data = tensor.tobytes()
    header = struct.pack('II', rows, cols)  # Pack dimensions as header
    print (len(data))
    print (len(struct.pack('ii', rows, cols)))
    # return struct.pack('ii', rows, cols) + data
    return header, data

def concatenate_tensors(tensors):
    # Determine the shape of the concatenated tensor
    num_rows = tensors[0].shape[0]
    num_cols = sum(tensor.shape[1] for tensor in tensors)

    # Create an empty array for the concatenated tensor
    concatenated_tensor = np.empty((num_rows, num_cols), dtype=np.float32)

    # Concatenate tensors row-wise
    col_start = 0
    for tensor in tensors:
        num_cols_tensor = tensor.shape[1]
        concatenated_tensor[:, col_start:col_start + num_cols_tensor] = tensor
        col_start += num_cols_tensor

    return concatenated_tensor

try:
    while True:
        # Example condition: send tensors every 5 seconds
        time.sleep(5)

        # Concatenate tensors
        concatenated_tensor = concatenate_tensors(tensors)

        # Serialize the concatenated tensor
        header, data = serialize_tensor(concatenated_tensor)

        # Send the serialized tensor
        total_size = len(header) + len(data)
        print (total_size)
        
        num_chunks = (total_size + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        # Send the length of the tensor data and number of chunks
        server_socket.sendto(struct.pack('II', total_size, num_chunks), (CLIENT_IP, PORT))
    
        # Send the header first
        server_socket.sendto(header, (CLIENT_IP, PORT))
        
        # Send data in chunks
        for i in range(num_chunks):
            start = i * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, total_size)
            print (start, end)
            server_socket.sendto(data[start:end], (CLIENT_IP, PORT))

        print("Tensor data sent in chunks.")

        # server_socket.sendto(struct.pack('!I', length), (CLIENT_IP, PORT))  # Replace with client's IP
        # server_socket.sendto(serialized_tensor, (CLIENT_IP, PORT))  # Replace with client's IP

        # Wait for acknowledgment from the client
        print("Waiting for acknowledgment from the client...")
        server_socket.settimeout(10)  # Wait for up to 10 seconds
        try:
            ack, addr = server_socket.recvfrom(1024)  # Buffer size of 1024 bytes for acknowledgment
            if ack.decode() == 'ACK':
                print("Acknowledgment received. Stopping server.")
                break
        except socket.timeout:
            print("No acknowledgment received. Retrying...")

except KeyboardInterrupt:
    print("Server shutting down...")

finally:
    server_socket.close()


