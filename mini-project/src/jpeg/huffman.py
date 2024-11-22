import heapq
from collections import defaultdict, namedtuple

class Node(namedtuple("Node", ["left", "right"])):
    def walk(self, code, acc):
        self.left.walk(code, acc + "0")
        self.right.walk(code, acc + "1")

class Leaf(namedtuple("Leaf", ["char"])):
    def walk(self, code, acc):
        code[self.char] = acc or "0"

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

def create_frequency_table(data):
    freq = defaultdict(int)
    for i in range(0, len(data), 2):
        DC = data[i]
        AC = data[i+1]
        freq[DC] += 1
        for value in AC:
            freq[value] += 1
    return freq

def huffman_code_tree(tree):
    code = {}
    tree.walk(code, "")
    return code

def huffman_encode(data, code):
    bitstream = ""
    for i in range(0, len(data), 2):
        DC = data[i]
        AC = data[i+1]
        
        encoded_DC = code[DC]
        bitstream += encoded_DC
        
        for j in range(len(AC)):
            bitstream += code[AC[j]]
            
            if AC[j] == 0 and j + 1 < len(AC) and AC[j+1] == 0:
                bitstream += code[0]  # Add the second 0
                break
    
    return bitstream

def huffman_decode(bitstream, code):
    reverse_code = {v: k for k, v in code.items()}
    decoded_data = []
    
    i = 0
    while i < len(bitstream):
        buffer = ""
        while buffer not in reverse_code:
            buffer += bitstream[i]
            i += 1
        DC = reverse_code[buffer]
        decoded_data.append(DC)
        
        AC = []
        buffer = ""
        while True:
            buffer += bitstream[i]
            i += 1
            if buffer in reverse_code:
                value = reverse_code[buffer]
                AC.append(value)
                buffer = ""
                if value == 0 and len(AC) >= 2 and AC[-2] == 0:
                    break
        decoded_data.append(AC)
    
    return decoded_data

def huff(data):
    freq_table = create_frequency_table(data)

    huff_tree = huffman_tree(freq_table)
    huff_codes = huffman_code_tree(huff_tree)
    bitstream = huffman_encode(data, huff_codes)
    return bitstream, huff_codes

def dehuff(bitstream, huff_codes):
    return huffman_decode(bitstream, huff_codes)


if __name__ == "__main__":
    data = [
        556, [1, 2, 3, 4, 0, 0],  # DC = 15, AC = [1, 2, 3, 0, 0, 4]
        18, [0, 1, 0, 5, 0, 0],  # DC = 18, AC = [0, 1, 0, 5]
    ]
    freq_table = create_frequency_table(data)
    huff_tree = huffman_tree(freq_table)
    huff_codes = huffman_code_tree(huff_tree)
    bitstream = huffman_encode(data, huff_codes)

    print("Encoded Bitstream:", bitstream)

    decoded_data = huffman_decode(bitstream, huff_codes)

    print("Decoded Data:", decoded_data)

