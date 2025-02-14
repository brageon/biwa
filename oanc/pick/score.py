import re, pickle

class Translator:
    def __init__(self, pickle_file):
        with open(pickle_file, "rb") as abc:
            self.tagger = pickle.load(abc)
            
    def remove_non(self, text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)

    def get_pos_tags(self, text):
        words = self.remove_non(text)
        return list(self.tagger.span_tokenize(words))
    
    def translate(self):
        with open('oak.txt', 'r') as file:
            lines = file.readlines()
        results = []
        for line_num, line in enumerate(lines):
            rex = line_num + 1
            wco = len(line.split())
            word_spans = self.get_pos_tags(line)
            if len(word_spans) == 1:
                start, end = word_spans[0]
                differences = [end]  
            else:
                differences = []
                prev_end = 0
                for i, tag in enumerate(word_spans):
                    differences.append(int(tag[0]) - prev_end)
                    prev_end = int(tag[0])
            avg_difference = sum(differences) / len(differences) if differences else 0
            rest =  round(int(avg_difference) / wco, 2)
            results.append((rex, rest))
        results.sort(key=lambda x: x[1])
        print(results)
            
path = 'english.pickle'
tran = Translator(path)
tran.translate()