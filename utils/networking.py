import socket
import struct
import numpy as np
import time

class Networking:
    def __init__(self, HOST="0.0.0.0", PORT=8080, CLIENT_IP="192.168.10.2"):
        self.HOST = HOST
        self.PORT = PORT
        self.CLIENT_IP = CLIENT_IP
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.HOST, self.PORT))
        self.server_socket.listen(1)

        print(f"Server listening on {self.HOST}:{self.PORT}")
        
        self.conn, self.addr = self.server_socket.accept()
        print(f"Connected by {self.addr}")

        # # Example tensor data using NumPy for demonstration
        # N = 50000
        # self.sample_tensors = [
        #     np.random.randn(N, 3).astype(np.float32),  # Tensor with 10 rows and 5 columns
        #     np.random.randn(N, 3).astype(np.float32),  # Tensor with 10 rows and 3 columns
        #     np.random.randn(N, 4).astype(np.float32),   # Tensor with 10 rows and 7 columns
        #     np.random.randn(N, 45).astype(np.float32),  # Tensor with 10 rows and 5 columns
        #     np.random.randn(N, 4).astype(np.float32),  # Tensor with 10 rows and 3 columns
        #     np.random.randn(N, 1).astype(np.float32)   # Tensor with 10 rows and 7 columns
  
        # ]

        self.sample_tensors = [
            np.random.randn(100000, 30).astype(np.float32),
            np.random.randn(100000, 15).astype(np.float32),
            np.random.randn(100000, 15).astype(np.float32)
        ]

    def __del__(self):
        self.conn.close()
        self.server_socket.close()

    def serialize_tensor(self, tensor):
        rows, cols = tensor.shape
        data = tensor.tobytes()
        return struct.pack('ii', rows, cols) + data

    def concatenate_tensors(self, tensors):        
        # Determine the shape of the concatenated tensor
        for tensor in tensors:
            # print (tensor)
            print (tensor.shape)
        num_rows = tensors[0].shape[0]
        num_cols = sum(tensor.shape[1] for tensor in tensors)

        # Create an empty array for the concatenated tensor
        concatenated_tensor = np.empty((num_rows, num_cols), dtype=np.float32)
        
        # Concatenate tensors row-wise
        col_start = 0
        for tensor in tensors:
            print (tensor)
            print (tensor.shape)
            num_cols_tensor = tensor.shape[1]
            concatenated_tensor[:, col_start:col_start + num_cols_tensor] = tensor
            col_start += num_cols_tensor
        return concatenated_tensor
        
    def send(self, should_send, tensors=None):
        if tensors is None:
            print ("[WARNING] Using default sample tensors for testing !!!\n")
            tensors = self.sample_tensors
            print ("Tensor 1: \n", self.sample_tensors[0])
            print ()
            print ("Tensor 2: \n", self.sample_tensors[1])
            print ()
            print ("Tensor 3: \n", self.sample_tensors[2])
            print ()
        while (should_send):
            print ("Sending tensors...")

            # Concatenate tensors
            # concatenated_tensor = self.concatenate_tensors(tensors)
            # print ("Tensor shape: ", concatenated_tensor.shape)

            # Serialize the concatenated tensor
            # serialized_tensor = self.serialize_tensor(concatenated_tensor)
            serialized_tensor = self.serialize_tensor(tensors)


            # Send the serialized tensor
            total_size = len(serialized_tensor)
            chunk_size = 5 * 1024 * 1024
            num_chunks = (total_size + chunk_size - 1) // chunk_size  # Calculate number of chunks

            start_time = time.time()  # Start timing the packet fly time

            self.conn.sendall(struct.pack('!I', total_size))
            self.conn.sendall(struct.pack('!I', num_chunks))
           
           # Send each chunk
            for i in range(num_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, total_size)
                self.conn.sendall(serialized_tensor[start:end])

            end_time = time.time()  # End timing the packet fly time
            print(f"Data transmission time: {end_time - start_time:.2f} seconds")
    
            # Wait for acknowledgment from the client
            print("Waiting for acknowledgment from the client...")
            self.conn.settimeout(10)  # Wait for up to 10 seconds
            try:
                ack = self.conn.recv(1024)  # Buffer size of 1024 bytes for acknowledgment
                if ack.decode() == 'ACK':
                    print("Acknowledgment received. Stopping server.")
                    should_send = False
                    break
            except socket.timeout:
                print("No acknowledgment received. Retrying...")
                should_send = True
        return should_send
