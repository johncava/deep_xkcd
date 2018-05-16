import glob
from PIL import Image
from collections import Counter

def get_list(num):
    items = []
    with open('data.txt') as f:
        for line in f:
            items.append(line.strip())
    return items[-num:]

#print get_list(20)
def read_data(comics):
    data = []
    character_corpus = []
    for comic in comics:
        strip_data = []
        directory = "xkcd_archive/" + str(comic)
        img_file = glob.glob(directory + "/*.png")
        if len(img_file) < 1:
            continue
        # Read Image
        img = Image.open(directory + "/" + img_file[0].split("/")[-1])
        img = img.resize((224,224), Image.ANTIALIAS)
        # IMPORTANT NOTE: LET BAND EQUAL 0 TO AVOID TUPLES IN THE IMAGE DATA
        img_data = list(img.getdata(band = 0))
        w, h = img.size
        img_data = [img_data[i * w :(i + 1) * w] for i in xrange(h)]
        # Read Transcript
        transcript = ""
        with open(directory + "/" + "xkcd-transcript-" + str(comic) + ".txt") as transcript_file:
            for line in transcript_file:
                transcript = transcript + line.strip() + " "
                for character in line:
                    if character == '\n':
                        continue
                    character_corpus = character_corpus + [character.lower()]
        data.append([img_data, transcript])
    print "Data is read"
    return data, character_corpus

def read_region_data(comics):
    data = []
    for comic in comics:
        strip_data = []
        directory = "xkcd_archive/" + str(comic)
        img_file = glob.glob(directory + "/*.png")
        if len(img_file) < 1:
            continue
        # Read Image
        img = Image.open(directory + "/" + img_file[0].split("/")[-1])
        img = img.resize((512,512), Image.ANTIALIAS)
        img_data = list(img.getdata(band = 0))
        w, h = img.size
        img_data = [img_data[i * w :(i + 1) * w] for i in xrange(h)]
        # Read region data
        regions = []
        with open(directory + '/region.data','r') as f:
            for line in f:
                r = line.strip("\n").split(",")
                r = [int(x) for x in r]
                regions.append(r)
        data.append([img_data,regions])
    print "Data is read"
    return data

def create_hot(dic_keys):
    hot = {}
    for index, key in enumerate(dic_keys):
        encode = [0.0] * (len(dic_keys) + 2)
        encode[index] = 1.0
        hot[key] = encode
    sos = [0.0] * (len(dic_keys) + 2)
    sos[(len(dic_keys))] = 1.0
    eos = [0.0] * (len(dic_keys) + 2)
    eos[(len(dic_keys) + 1)] = 1.0
    hot['<SOS>'] = sos
    hot['<EOS>'] = eos
    return hot
        
#data, corpus = read_data(get_list(20))
#dictionary = Counter(corpus)
#d = create_hot(dictionary.keys())
