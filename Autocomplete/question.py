from nltk.corpus import gutenberg

with open('../data/emma.txt', 'w') as f:
    f.write(gutenberg.raw('austen-emma.txt'))

# text = open('../data/game_of_thrones.txt', encoding='utf-8')

# for paragraph in text:
#     paragraph = paragraph.lower()
#     print(paragraph)