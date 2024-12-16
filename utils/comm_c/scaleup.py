import socket
import struct
import numpy as np
import time

# Define the IP address and port to communicate with the client
HOST = '0.0.0.0'
PORT = 8080

# Create a TCP socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"Server listening on {HOST}:{PORT}")

# Accept a connection
conn, addr = server_socket.accept()
print(f"Connected by {addr}")

# Example tensor data using NumPy for demonstration
tensors = [
    np.random.randn(100000, 30).astype(np.float32),
    np.random.randn(100000, 15).astype(np.float32),
    np.random.randn(100000, 15).astype(np.float32)
]

print ("Tensor 1: \n", tensors[0])
print ()
print ("Tensor 2: \n", tensors[1])
print ()
print ("Tensor 3: \n", tensors[2])
print ()

def serialize_tensor(tensor):
    rows, cols = tensor.shape
    data = tensor.tobytes()
    return struct.pack('ii', rows, cols) + data

def concatenate_tensors(tensors):
    num_rows = tensors[0].shape[0]
    num_cols = sum(tensor.shape[1] for tensor in tensors)
    concatenated_tensor = np.empty((num_rows, num_cols), dtype=np.float32)
    col_start = 0
    for tensor in tensors:
        num_cols_tensor = tensor.shape[1]
        concatenated_tensor[:, col_start:col_start + num_cols_tensor] = tensor
        col_start += num_cols_tensor
    return concatenated_tensor

try:
    while True:
        time.sleep(5)
        
        start_time = time.time()  # Start
        
        concatenated_tensor = concatenate_tensors(tensors)
        serialized_tensor = serialize_tensor(concatenated_tensor)

        # Send the length of the tensor data
        total_size = len(serialized_tensor)
        chunk_size = 5 * 1024 * 1024
        num_chunks = (total_size + chunk_size - 1) // chunk_size  # Calculate number of chunks


        trans_start_time = time.time()  # Start timing the packet fly time
        print(f"Data preprocessing time: {trans_start_time - start_time:.2f} seconds")

        conn.sendall(struct.pack('!I', total_size))
        conn.sendall(struct.pack('!I', num_chunks))

        # Send each chunk
        for i in range(num_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, total_size)
            conn.sendall(serialized_tensor[start:end])

        end_time = time.time()  # End timing the packet fly time
        print(f"Data transmission time: {end_time - trans_start_time:.2f} seconds")

        # Wait for acknowledgment from the client
        print("Waiting for acknowledgment from the client...")
        conn.settimeout(10)  # Wait for up to 10 seconds
        try:
            ack = conn.recv(1024)
            if ack.decode() == 'ACK':
                print("Acknowledgment received. Stopping server.")
                # break
        except socket.timeout:
            print("No acknowledgment received. Retrying...")

except KeyboardInterrupt:
    print("Server shutting down...")

finally:
    conn.close()
    server_socket.close()
