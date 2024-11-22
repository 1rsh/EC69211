import struct
import numpy as np

def bits_to_bytes(bit_string):
    """Convert a string of 0s and 1s to a bytes object."""
    # Preserve the exact bit length as meta-information to avoid padding mismatches
    bit_length = len(bit_string)
    byte_length = (bit_length + 7) // 8
    byte_value = int(bit_string, 2).to_bytes(byte_length, byteorder='big')
    return byte_value, bit_length

def bytes_to_bits(byte_string, bit_length):
    """Convert a bytes object back to a string of 0s and 1s."""
    # Preserve the exact original length of the bit string
    bit_string = ''.join(f"{byte:08b}" for byte in byte_string)
    return bit_string[-bit_length:]  # Only keep the relevant bits

# def save_to_file(data, filename):
#     with open(filename, 'wb+') as file:
#         file.write(struct.pack('>I', len(data)))  # Number of entries
#         for bit_string, dictionary in data:
#             byte_string, bit_length = bits_to_bytes(bit_string)
#             dict_str = ','.join(f"{key}:{value}" for key, value in dictionary.items())
#             dict_bytes = dict_str.encode('utf-8')
            
#             file.write(struct.pack('>I', len(byte_string)))  # Length of byte string
#             file.write(struct.pack('>I', bit_length))       # Length of original bit string
#             file.write(struct.pack('>I', len(dict_bytes)))  # Length of dictionary bytes
#             file.write(byte_string)
#             file.write(dict_bytes)

# def load_from_file(filename):
#     data = []
#     with open(filename, 'rb') as file:
#         for i in range(3):
#             file.readline()
#         entry_count = struct.unpack('>I', file.read(4))[0]  # Number of entries
        
#         for _ in range(entry_count):
#             byte_string_len = struct.unpack('>I', file.read(4))[0]  # Length of byte string
#             bit_length = struct.unpack('>I', file.read(4))[0]       # Length of original bit string
#             dict_bytes_len = struct.unpack('>I', file.read(4))[0]   # Length of dictionary bytes
            
#             byte_string = file.read(byte_string_len)  # Read byte string
#             dict_bytes = file.read(dict_bytes_len)    # Read dictionary bytes
            
#             bit_string = bytes_to_bits(byte_string, bit_length)  # Reconstruct bit string
#             dictionary = {int(item.split(':')[0]): item.split(':')[1] for item in dict_bytes.decode('utf-8').split(',')}
            
#             data.append((bit_string, dictionary))
    
#     return data

def save_to_file(data, filename, extra_tuple=None):
    with open(filename, 'wb') as file:
        file.write(struct.pack('>I', len(data)))  # Number of entries
        for bit_string, dictionary in data:
            byte_string, bit_length = bits_to_bytes(bit_string)
            dict_str = ','.join(f"{key}:{value}" for key, value in dictionary.items())
            dict_bytes = dict_str.encode('utf-8')
            
            file.write(struct.pack('>I', len(byte_string)))  # Length of byte string
            file.write(struct.pack('>I', bit_length))       # Length of original bit string
            file.write(struct.pack('>I', len(dict_bytes)))  # Length of dictionary bytes
            file.write(byte_string)
            file.write(dict_bytes)
        
        if extra_tuple:
            file.write(struct.pack('>5I', *extra_tuple))  # Save the extra tuple as three unsigned integers

def load_from_file(filename):
    data = []
    extra_tuple = None
    with open(filename, 'rb') as file:
        entry_count = struct.unpack('>I', file.read(4))  # Number of entries
        
        for _ in range(entry_count[0]):
            byte_string_len = struct.unpack('>I', file.read(4))[0]  # Length of byte string
            bit_length = struct.unpack('>I', file.read(4))[0]       # Length of original bit string
            dict_bytes_len = struct.unpack('>I', file.read(4))[0]   # Length of dictionary bytes
            
            byte_string = file.read(byte_string_len)  # Read byte string
            dict_bytes = file.read(dict_bytes_len)    # Read dictionary bytes
            
            bit_string = bytes_to_bits(byte_string, bit_length)  # Reconstruct bit string
            dictionary = {int(item.split(':')[0]): item.split(':')[1] for item in dict_bytes.decode('utf-8').split(',')}
            
            data.append((bit_string, dictionary))
        
        remaining_data = file.read()  # Read any remaining data
        if remaining_data:
            extra_tuple = struct.unpack('>5I', remaining_data[:20])  # Read the 3-integer tuple
    
    return data, extra_tuple


if __name__ == "__main__":
    to_save = [('0101101000101010010100001111100', {1: '0', 2: '10', np.int64(-3): '11', 4: '110'})]*3
    save_to_file(to_save, 'cache.myjpeg', (426, 640, 3, 50, 0))
    loaded = load_from_file('cache.myjpeg')
    print(to_save)
    print(loaded)

    print(to_save == loaded)