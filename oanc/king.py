import re, spacy, pickle

class BaCl:
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
        seconds = [tuple[1] for tuple in results]
        return seconds
            
if __name__ == "__main__":
    path = './pick/english.pickle'
    tran = BaCl(path)
    
class GaFe:
    def __init__(self):
        self.gaf = tran.translate()
        self.nlp = spacy.load('en_core_web_sm')   
    def yuno(self):
        dat = self.gaf
        for value in dat:
            return value         
        
    def remove_non(self, text):
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    def get_pos_tags(self, text):
        words = self.remove_non(text).split()
        if words:
            doc = self.nlp(" ".join(words))
            return [(token.text, token.tag_) for token in doc]
        else:
            return []
            
    def classify_tag(self, tag):
        tag_to_classify = tag[1] 
        tag_map = { 'NNP': 'Z', 'NNPS': 'Z', 'NN': 'Z', 'NNS': 'Z',  # Nouns (proper and common)
        'WP': 'Z', 'WPS': 'Z', 'DT': 'Z', 'PDT': 'Z', 'WDT': 'Z',  # Pronouns, determiners, wh-determiners
        'PRP': 'A', 'PRPS': 'A', 'PRP$': 'A',  # Pronouns (personal, possessive)
        'EX': 'A', 'POS' : 'A', 'TO': 'A', 'UH': 'J',  # Existential, adpositions, conjunctions, interjections
        'IN': 'J', 'CD': 'J', 'FW': 'J', 'RP': 'J', 'CC': 'J', 'MD' : 'J',  # Prepositions, foreign words, particles, modal verbs
        'JJ': 'T', 'JJR': 'T', 'JJS': 'T',  # Adjectives (positive, comparative, superlative)
        'RB': 'T', 'WRB': 'T', 'RBR': 'T', 'RBS': 'T',  # Adverbs (positive, comparative, superlative)
        'VBP': 'G', 'VBZ': 'G', 'VBD': 'G', 'VBG': 'G', 'VBN': 'G', 'VB': 'G' }
        return tag_map.get(tag_to_classify)
        
    def hicks_tag(self, word_spans):
        duck = r"G Z | Z G | T J | J T"
        youn = r"A J T | G G Z | J A T | J J T | A Z T | Z J Z | G Z G | G A T | Z J G | G Z J | T G A | J Z Z | Z Z Z | T A G | A G T | T J Z | T A T | Z J T | T Z J | J Z T | Z Z T | T G G | G J T | T G J | T Z G | T Z Z | J G T | Z G T | G G T | T G Z | T J T | G Z T | T Z T"
        matches = re.findall(youn, word_spans)
        return len(duck), len(matches)
        
    def mine(self):
        anchors = [18/17, 18/16, 18/15, 18/14, 18/13, 18/12, 18/11, 18/10, 18/9, 18/8, 18/7, 18/6, 18/5, 18/4, 18/3, 18/2]
        letters = ['DN', 'ND', 'DC', 'CD', 'DD', 'CH', 'HC', 'CC', 'NC', 'CN', 'NN', 'NH', 'HN', 'HD', 'DH', 'HH']
        with open('oak.txt', 'r') as file:
            lines = file.readlines()
        for num, line in enumerate(lines):
            rex = num + 1
            wco = len(line.split()) * 2
            spacy_output = self.get_pos_tags(line)
            word_spans = [self.classify_tag(tag) for tag in spacy_output]
            word_spans = " ".join(word_spans)
            duck, hicks = self.hicks_tag(word_spans)
            dual = duck / wco
            val = self.yuno()
            dual = round(val * dual, 2)
            closest_index = min(range(len(anchors)), key=lambda i: abs(anchors[i] - dual))
            anchor = letters[closest_index]
            print("{:02d}".format(rex), f"{dual:.2f}", f"{anchor}")
              
bacl = GaFe()
drive = bacl.mine()
