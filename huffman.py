import numpy as np
import bisect
from PIL import Image
import collections
import matplotlib.pyplot as plt

# 2. Algorithme de Huffman

class Node:
    def __init__(self, l=None, r=None, i=0, p=0):
        self.left = l
        self.right = r
        self.prob = p
        self.key = i

    def print(self, k=0):
        print("---------------\nNODE\nprob=" + str(self.prob))
        
        if (self.right is not None):
            print("\nRIGHT\n prob=" + str(self.right.prob) + " key=" + str(self.right.key))

        if (self.left is not None):
            print("\nLEFT\n prob=" + str(self.left.prob) + " key=" + str(self.left.key))

        if (k == 0):
            if (self.right is not None):
                print("---------------\nRIGHT " + str(self.prob))
                self.right.print()

            if (self.left is not None):
                print("---------------\nLEFT " + str(self.prob))
                self.left.print()

    def is_leaf(self):
        return self.left is None and  self.right is None

    def is_none(self):
        return self is None

class Node2:
    def __init__(self, l=None, r=None, i=0, c=0):
        self.left = l
        self.right = r
        self.code = c
        self.key = i

    def print(self, k=0):
        print("---------------\nNODE\ncode=" + str(self.code))
        
        if (self.right is not None):
            print("\nRIGHT\n code=" + str(self.right.code) + " key=" + str(self.right.key))

        if (self.left is not None):
            print("\nLEFT\n code=" + str(self.left.code) + " key=" + str(self.left.key))

        if (k == 0):
            if (self.right is not None):
                print("---------------\nRIGHT " + str(self.code))
                self.right.print()
                
            if (self.left is not None):
                print("---------------\nLEFT " + str(self.code))
                self.left.print()

    def is_leaf(self):
        return self.left is None and  self.right is None

    def is_none(self):
        return self is None
    
def huffman_tree(proba):
    if (len(proba) <= 0):
        return Node()
    else:
        proba_sorted = sorted(proba)
        nodes = []

        for i in range(len(proba_sorted)):
            nodes.append(Node(p=proba_sorted.pop(0)))
        
        while (len(nodes) > 1):
            n1 = nodes.pop(0)
            n2 = nodes.pop(0)
            n = Node()

            if (n1.prob < n2.prob):
                n1.key = 1
                n.right = n1
                n.left = n2
            else:
                n2.key = 1
                n.right = n2
                n.left = n1
                
            n.prob = n1.prob + n2.prob    
            nodes.append(n)
            nodes = sorted(nodes, key=lambda node:node.prob)

        return nodes.pop(0)

def get_cwd(tree):
    if (tree.is_none()):
        return []
    else:
        if (tree.is_leaf()):
            code = {"key":str(tree.key), "proba":tree.prob}
            return [code]
        else:
            codewords_l = get_cwd(tree.left)
            codewords_r = get_cwd(tree.right)
            codewords = np.concatenate((codewords_r, codewords_l), axis=0)

            if (tree.prob != 1):
                for i in range(len(codewords)):
                    codewords[i]["key"] = str(tree.key) + codewords[i]["key"]
        
        return codewords

def huffman_code(proba):
    tree = huffman_tree(proba)
    codewords = get_cwd(tree)
    lgth = 0

    for code in codewords:
        lgth += len(code["key"]) * code["proba"]
        
    return (codewords, lgth)
    
def huffman_tree2(codewords):
    if (len(codewords) < 0):
        return Node2()
    else:
        nodes = []
        cwd = sorted(codewords, key=lambda code:len(code), reverse=True)

        for i in range(len(cwd)):
            nodes.append(Node2(c=cwd.pop(0)))
            
        while (len(nodes) > 1):
            if (len(nodes[0].code) < len(nodes[1].code)):
                nodes = sorted(nodes, key=lambda node:(len(node.code), int(node.code[-1])), reverse=True)
                
            n1 = nodes.pop(0)
            n2 = nodes.pop(0)
            n = Node2()

            if (n1.code[len(n1.code) - 1] == '1'):
                n1.key = 1
                n.right = n1
                n.left = n2
            else:
                n2.key = 1
                n.right = n2
                n.left = n1

            n.code = n1.code[:len(n1.code) - 1]
            nodes.insert(0, n)
                    
        return nodes.pop(0)

def cwd_detect(tree, seq):
    cwd = ""
    n = tree
    i = 0

    for j in range(len(seq)):
        n = n.right if (seq[j] == '1') else n.left
        i += 1
        cwd += str(n.key)
        
        if (n.is_leaf()):
            return (i, cwd)
                    
    return (i, "")    
    
def huffman_decode(seq, symb, codewords):
    tree = huffman_tree2(codewords)
    sequence = seq
    msg = ""
    
    while (len(sequence) > 0):
        i, cwd = cwd_detect(tree, sequence)
        sequence = sequence[i:]
        msg += symb[codewords.index(cwd)]
                
    return msg

# 3. Experimentations

def get_img_values(datas):
    counter = collections.Counter(datas)
    values = counter.values()
    return values

def get_data_frequencies(data_length, values):
    frequencies = [val / data_length for val in values]
    return frequencies

def histogram(datas):
    height = np.histogram(datas, density=True, bins =256)[0]
    width = np.histogram(datas, density = True, bins = 256)[1][0:len(height)]
    plt.bar(width, height)
    plt.show()
    return height


def compute_entropie(proba):
    entropie_exp = 0
    
    for p in proba:
        if p !=0:
            entropie_exp -= p * np.log(p)
        
    return entropie_exp

def compute_nb_bits_compressed(values, frequencies, codewords_exp):
    vals = list(values)
    values_freq = []
    
    for i in range (len(vals)):
        val_freq = {"mot":vals[i], "proba":frequencies[i]}
        values_freq.append(val_freq)

    vals_img_compressed = ""
    for i in range(len(values_freq)):
        j = 0
        while j < len(codewords_exp):
            if (values_freq[i]["proba"] == codewords_exp[j]["proba"]):
                vals_img_compressed += codewords_exp[i]["key"]
                break
            j += 1
        

    nb_bits_compressed = len(vals_img_compressed)
    return nb_bits_compressed

def experimentation_img(img_name, nb_bits_uncompressed):
    print("-- Experimentations image '" + img_name + "' --")
    # Recuperation des donnees du fichier
    img = list(Image.open(img_name).getdata())
        
    # Longueur moyenne
    values = get_img_values(img)
    frequencies = get_data_frequencies(len(img), values)
    codewords_exp, lgth_exp = huffman_code(frequencies)
    print("longueur moyenne = " + str(lgth_exp))

    # Histogramme
    probas = histogram(img)
    
    # Entropie
    entropie_exp = compute_entropie(probas)
    print("entropie experimentale = " + str(entropie_exp))

    # Ratio de compression
    nb_bits_compressed = compute_nb_bits_compressed(values, frequencies, codewords_exp)
    ratio = nb_bits_compressed / nb_bits_uncompressed
    print("ratio de compression = " + str(ratio) + "\n")

def experimentation_img(img_name, nb_bits_uncompressed):
    print("-- Experimentations image '" + img_name + "' --")
    # Recuperation des donnees du fichier
    img = list(Image.open(img_name).getdata())
        
    # Longueur moyenne
    values = get_img_values(img)
    frequencies = get_data_frequencies(len(img), values)
    codewords_exp, lgth_exp = huffman_code(frequencies)
    print("longueur moyenne = " + str(lgth_exp))

    # Histogramme
    probas = histogram(img)
    
    # Entropie
    entropie_exp = compute_entropie(probas)
    print("entropie experimentale = " + str(entropie_exp))

    # Ratio de compression
    nb_bits_compressed = compute_nb_bits_compressed(values, frequencies, codewords_exp)
    ratio = nb_bits_compressed / nb_bits_uncompressed
    print("ratio de compression = " + str(ratio) + "\n")

def experimentation_img(img_name, nb_bits_uncompressed):
    print("-- Experimentations image '" + img_name + "' --")
    
    # Recuperation des donnees du fichier
    img = list(Image.open(img_name).getdata())
        
    # Longueur moyenne
    values = get_img_values(img)
    frequencies = get_data_frequencies(len(img), values)
    codewords_exp, lgth_exp = huffman_code(frequencies)
    print("\tLongueur moyenne = " + str(lgth_exp))

    # Histogramme
    probas = histogram(img)
    # Entropie
    entropie_exp = compute_entropie(probas)
    print("\tEntropie experimentale = " + str(entropie_exp))

    # Ratio de compression
    nb_bits_compressed = compute_nb_bits_compressed(values, frequencies, codewords_exp)
    ratio = nb_bits_compressed / nb_bits_uncompressed
    print("\tRatio de compression = " + str(ratio) + "\n")

def experimentation_txt(txt_name, nb_bits_uncompressed):
    print("-- Experimentations texte '" + txt_name + "' --")
    
    # Recuperation des donnees du fichier
    txt = list(open(txt_name, 'r').read())

    # Table de frequence des caracteres
    values = get_img_values(txt)
    frequencies = get_data_frequencies(len(txt), values)
    print("\tTable de frequence des caracteres = ")
    print(frequencies)
    
    # # Longueur moyenne
    codewords_exp, lgth_exp = huffman_code(frequencies)
    print("\tLongueur moyenne = " + str(lgth_exp))

    # Entropie
    # probas = histogram(txt)
    # entropie_exp = compute_entropie(probas)
    # print("\tEntropie experimentale = " + str(entropie_exp))

    # # Ratio de compression
    nb_bits_compressed = compute_nb_bits_compressed(values, frequencies, codewords_exp)
    ratio = nb_bits_compressed / nb_bits_uncompressed
    print("\tRatio de compression = " + str(ratio) + "\n")

def main():
    # 2 - Algorithme de Huffman

    # Encodage
    proba = [0.5, 0.26, 0.11, 0.04, 0.04, 0.03, 0.01, 0.01 ]
    proba = [0.125, 0.125, 0.25, 0.5 ]
    codewords_enc, lgth = huffman_code(proba)

    # AFFICHAGES TESTS ENCODAGE
    #print("-------ENCODAGE-------\nproba=")
    #print(proba)
    #print("codewords=")
    #print(codewords_enc)
    #print("lgth=" + str(lgth))

    
    # Decodage
    symb = ['a', 'b', 'c', 'e', 'f', 'g', 'h', 'i']
    codewords_dec = ["0", "10", "111", "11000", "11001", "11010", "110110", "110111"]
    seq = "011011110"
    msg = huffman_decode(seq, symb, codewords_dec)

    # AFFICHAGES TESTS DECODAGE
    print("\n-------DECODAGE-------\nsymb=")
    print(symb)
    print("codewords=")
    print(codewords_dec)
    print("seq=" + seq + "\nmsg=" + msg)

    
    
    # 3 - Experimentations

    # 1. Compression d'images
    #experimentation_img('C:/Users/LENOVO/Downloads/goldhill.png', 173024)
    #experimentation_img('C:/Users/LENOVO/Downloads/boat.png', 177762)
    #experimentation_img('C:/Users/LENOVO/Downloads/moon.png', 75294)
    
    # 2. Compression de texte
    #experimentation_txt('C:/Users/LENOVO/Downloads/buscon.txt', 245469)
    #experimentation_txt('C:/Users/LENOVO/Downloads/clair-de-lune.txt', 157338)
    #experimentation_txt('C:/Users/LENOVO/Downloads/dorian.txt', 445879)    
               
if __name__ == "__main__":
    main()
