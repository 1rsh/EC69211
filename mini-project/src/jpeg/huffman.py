import heapq
from collections import defaultdict, namedtuple

# A node for the Huffman Tree
class Node(namedtuple("Node", ["left", "right"])):
    def walk(self, code, acc):
        self.left.walk(code, acc + "0")
        self.right.walk(code, acc + "1")

class Leaf(namedtuple("Leaf", ["char"])):
    def walk(self, code, acc):
        code[self.char] = acc or "0"

# Build a Huffman Tree
def huffman_tree(frequencies):
    heap = []
    for char, freq in frequencies.items():
        heap.append((freq, len(heap), Leaf(char)))
    heapq.heapify(heap)

    count = len(heap)
    while len(heap) > 1:
        freq1, _count1, left = heapq.heappop(heap)
        freq2, _count2, right = heapq.heappop(heap)
        heapq.heappush(heap, (freq1 + freq2, count, Node(left, right)))
        count += 1

    return heap[0][-1]

# Create a frequency dictionary from the data
def create_frequency_table(data):
    freq = defaultdict(int)
    for i in range(0, len(data), 2):
        DC = data[i]
        AC = data[i+1]
        freq[DC] += 1
        for value in AC:
            freq[value] += 1
    return freq

# Generate Huffman codes from the tree
def huffman_code_tree(tree):
    code = {}
    tree.walk(code, "")
    return code

# Encode data into a single bitstream
def huffman_encode(data, code):
    bitstream = ""
    for i in range(0, len(data), 2):
        DC = data[i]
        AC = data[i+1]
        
        # Encode DC
        encoded_DC = code[DC]
        bitstream += encoded_DC
        
        # Encode AC, knowing it ends with [0, 0]
        for j in range(len(AC)):
            bitstream += code[AC[j]]
            
            # If we encounter the terminating sequence 0, 0, break early
            if AC[j] == 0 and j + 1 < len(AC) and AC[j+1] == 0:
                bitstream += code[0]  # Add the second 0
                break
    
    return bitstream

# Decode bitstream back to original data
def huffman_decode(bitstream, code):
    reverse_code = {v: k for k, v in code.items()}
    decoded_data = []
    
    i = 0
    while i < len(bitstream):
        # Decode DC
        buffer = ""
        while buffer not in reverse_code:
            buffer += bitstream[i]
            i += 1
        DC = reverse_code[buffer]
        decoded_data.append(DC)
        
        # Decode AC
        AC = []
        buffer = ""
        while True:
            buffer += bitstream[i]
            i += 1
            if buffer in reverse_code:
                value = reverse_code[buffer]
                AC.append(value)
                buffer = ""
                # If we encounter the terminating 0, 0 sequence, stop
                if value == 0 and len(AC) >= 2 and AC[-2] == 0:
                    break
        decoded_data.append(AC)
    
    return decoded_data

def huff(data):
    freq_table = create_frequency_table(data)

    # Build Huffman Tree
    huff_tree = huffman_tree(freq_table)

    # Generate Huffman codes
    huff_codes = huffman_code_tree(huff_tree)

    # Huffman encode the data
    bitstream = huffman_encode(data, huff_codes)
    return bitstream, huff_codes

def dehuff(bitstream, huff_codes):
    return huffman_decode(bitstream, huff_codes)


# Main function to encode and decode
if __name__ == "__main__":
    data = [
        556, [1, 2, 3, 4, 0, 0],  # DC = 15, AC = [1, 2, 3, 0, 0, 4]
        18, [0, 1, 0, 5, 0, 0],  # DC = 18, AC = [0, 1, 0, 5]
    ]
    # Create frequency table
    freq_table = create_frequency_table(data)

    # Build Huffman Tree
    huff_tree = huffman_tree(freq_table)

    # Generate Huffman codes
    huff_codes = huffman_code_tree(huff_tree)

    # Huffman encode the data
    bitstream = huffman_encode(data, huff_codes)

    print("Encoded Bitstream:", bitstream)

    # Huffman decode the bitstream
    decoded_data = huffman_decode(bitstream, huff_codes)

    print("Decoded Data:", decoded_data)

