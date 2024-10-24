import struct
import numpy as np

def save_byte_strings(filename, byte_strings):
    """
    Save multiple byte strings to a file.
    
    Args:
        filename (str): The name of the file.
        byte_strings (bytes): Variable number of byte strings to save.
    """
    with open(filename, 'wb') as f:
        for byte_string in byte_strings:
            # First, store the length of the byte string, then the actual byte string
            f.write(len(byte_string).to_bytes(4, 'big'))  # Write length (4 bytes)
            f.write(byte_string)  # Write the actual byte string

def retrieve_byte_strings(filename):
    """
    Retrieve byte strings from a file.
    
    Args:
        filename (str): The name of the file.

    Returns:
        list: A list of retrieved byte strings.
    """
    byte_strings = []
    with open(filename, 'rb') as f:
        while True:
            # Read the length of the next byte string (4 bytes)
            length_bytes = f.read(4)
            if not length_bytes:
                break  # End of file
            length = int.from_bytes(length_bytes, 'big')
            
            # Read the byte string of the specified length
            byte_string = f.read(length)
            byte_strings.append(byte_string)

    return byte_strings

def rle_data_to_bytes(rle_data):
    """
    Convert the RLE encoded data into a byte stream.
    rle_data: List of integers (or tuples), representing RLE data.
    """
    byte_stream = bytearray()

    for item in rle_data:
        if isinstance(item, int) or isinstance(item, np.int64):
            # Convert the integer to 2 bytes (signed)
            byte_stream += struct.pack(">h", item)
        elif isinstance(item, list) and len(item) % 2 == 0:
            # Assuming the RLE data is in (run-length, value) pairs
            for i in zip(item[::2], item[1::2]):
                run_length, value = i
                try:
                    byte_stream += struct.pack(">bb", run_length, value)
                except Exception as e:
                    print(run_length, value)
                    raise Exception
        else:
            raise ValueError("Unexpected RLE format:", type(item))
    
    return bytes(byte_stream)

def bytes_to_rle_data(byte_stream):
    """
    Convert a byte stream back into RLE encoded data.
    byte_stream: A bytearray or bytes object representing the compressed RLE data.
    Returns a list of RLE data (integers or (run-length, value) pairs).
    """
    rle_data = []
    i = 0
    while i < len(byte_stream):
        if i + 1 < len(byte_stream):
            # Unpack as (run-length, value) pair (2 bytes: 1 byte for run-length and 1 byte for value)
            run_length, value = struct.unpack(">ib", byte_stream[i:i+5])
            rle_data.extend((run_length, value))
            i += 2
        else:
            # If a single value (without a pair), decode as a signed short (2 bytes)
            value = struct.unpack(">h", byte_stream[i:i+2])[0]
            rle_data.append(value)
            i += 2

    return rle_data



import struct

def bits_to_bytes(bit_string):
    # Convert a string of bits to bytes
    return bytes(int(bit_string[i:i + 8], 2) for i in range(0, len(bit_string), 8))

def bytes_to_bits(byte_string):
    # Convert bytes to a string of bits
    return ''.join(format(byte, '08b') for byte in byte_string)

def save_to_file(data, filename):
    with open(filename, 'wb') as file:
        # First write the number of entries as a big-endian integer
        file.write(struct.pack('>I', len(data)))  # >I for big-endian unsigned int
        for bit_string, dictionary in data:
            byte_string = bits_to_bytes(bit_string)
            dict_str = ','.join(f"{key}:{value}" for key, value in dictionary.items())
            dict_bytes = dict_str.encode('utf-8')
            
            # Write the length of byte_string and dict_bytes
            file.write(struct.pack('>I', len(byte_string)))  # Length of the bit string
            file.write(struct.pack('>I', len(dict_bytes)))   # Length of the dictionary
            
            # Write the actual bytes
            file.write(byte_string)
            file.write(dict_bytes)

def load_from_file(filename):
    data = []
    with open(filename, 'rb') as file:
        # Read the number of entries
        entry_count = struct.unpack('>I', file.read(4))[0]
        
        for _ in range(entry_count):
            # Read the length of byte_string and dict_bytes
            byte_string_len = struct.unpack('>I', file.read(4))[0]
            dict_bytes_len = struct.unpack('>I', file.read(4))[0]
            
            # Read the actual bytes
            byte_string = file.read(byte_string_len)
            dict_bytes = file.read(dict_bytes_len)
            
            # Convert bytes back to bit strings
            bit_string = bytes_to_bits(byte_string)
            # Convert the dictionary bytes back to a dictionary
            dictionary = {item.split(':')[0]: item.split(':')[1] for item in dict_bytes.decode('utf-8').split(',')}
            
            # Append the tuple to the data list
            data.append((bit_string, dictionary))
    
    return data