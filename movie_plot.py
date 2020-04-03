# make time series of distances from dialogue
import numpy as np
import bmd3 as bmd03

# process bytes into string, process string to fix artifacts
path = 'movie_lines.txt'
with open(path, 'rb') as f:
  text = f.readlines()
txt = [x.decode("utf-8", "replace") for x in text]
txts = [t.split("+++$+++")[-1].split("\n")[0] for t in txt]
txtss = []
for i, t in enumerate(txts):
    Lst = t.split("'")
    new_line = []
    for ele in Lst:
        new_line.append(ele[:-1])
    txtss.append("'".join(new_line))

# dictionary of meta data
path_meta = 'movie_titles_metadata.txt'
with open(path_meta, 'rb') as f:
  text_meta = f.readlines()
txt_meta = [x.decode("utf-8", "replace") for x in text_meta]#to remove /n?
txts_genres = {t.split("+++$+++")[0].strip() : t.split("+++$+++")[-1].strip().replace("[","").replace("]","").replace("'", "").split(", ") for t in txt_meta}
txts_names =  {t.split("+++$+++")[0].strip() : t.split("+++$+++")[1].strip() for t in txt_meta}
#print(txts_meta["m0"])
# ... "m0" : ['romance', 'thriller']
#for k,v in txts_names.items(): print(k,v)

#have meta data, now can label movie by genres/name
#choose a suitable reference description -- plot movie lines vs reference description in time-series. label each graph with the movie / genres


# for i,t in txtss[-10:]: print(i, t)
def parse_line(line):
    movie = line.split("+++$+++")[-3].strip()
    movie_line = line.split("+++$+++")[-1].split("\n")[0]
    # remove spaces before all apostraphes 
    movie_line_edited = "'".join([x for x in movie_line.split(" '")])
    return movie, movie_line

lines = [x.decode("utf-8", "replace") for x in text]
movie_dict = {}
for line in lines:
    movie, movie_line = parse_line(line)
    if movie in movie_dict:
        movie_dict[movie].append(movie_line)
    else:
        movie_dict[movie] = [movie_line]
#############################################################################################################################################################

# take k-sentence chunks as documents at time t and at time t+n, make BMD time series
from matplotlib import pyplot as plt
bmd_inst = bmd03.BertMoversDistance(512, normalized_dist_entries=True)
bert = bmd_inst.B

def BMD_series(reference_doc, list_of_docs, k): #list_of_docs: all lines in a given movie
    # combine k movie lines into one paragraph
    assert(k>=3)
    min_d = 10000
    mid_doc = ''
    output = []
    n_cut_docs = 0
    for i in range(len(list_of_docs)):
        if (i+1)%k == 0 and i>1:
            doc = ".".join(list_of_docs[i-k+1:i+1])
            tok = bmd_inst.B.tokenizer.tokenize(doc)
            if len(tok)>512: #512 is the upper limit on number of tokens that BERT can take as input
              ave = np.average([len(x) for x in tok]) 
              doc = doc[:int(ave)*511]        
              n_cut_docs += 1
            dd = bmd_inst(d1=doc, d2=ref_doc)
            output.append(dd)   
            if dd<min_d:
              min_d = dd
              mid_doc = doc
    print("Number of docs cut to 512: " + str(n_cut_docs))        
    return output, min_d, mid_doc

#ref_doc = "Science fiction speculative science-based depictions phenomena mainstream extraterrestrial lifeforms alien worlds extrasensory perception time travel futuristic elements spacecraft robots cyborgs interstellar travel technologies political social issues explore philosophical the human condition planets ships gravity lightspeed warp transporter lasers phasers galaxy people."
#ref_doc = "The woman crept down the hall in the dark, holding baseball bat above her head, her arms shaking, lungs begging for air as she tried not to breath. She knew it was here, the one who stalked her, terrorized her, tried to kill her—she didn’t care what everyone said. She knew it was all of it was real, and now she was going to catch it and prove to everyone that she wasn’t insane. She was almost at the doorway. 5 more steps. 4 more steps. 3. 2. 1…BANG! She threw open door, and swung the bat with all of the force in her being."
ref_doc = "As the teenage boy stepped into the old mansion, his friends cackling behind him, he thought he could hear things that, he forced himself to believe, were in his head—rattling bones, scurrying rats, hushed whispers…and the slow drip, drip, drip, coming from a spot he told himself wasn’t really there; the red, oozing stain in the ceiling boards above. He only had to spend one hour in the house and he would prove to his friends that he wasn’t afraid. Just one hour. He took one last glance out the door before shutting out the light of the full moon, enclosing himself in complete darkness, with only the sound of his racing, terrified thoughts."
def prin(movie,min_d, mid_doc):
  print("movie _"+str(movie)+"_")
  print("name _"+str(txts_names[movie])+"_")
  print("genres _"+str(txts_genres[movie])+"_")
  print("k-line document with min bmd of " + str(min_d) + " is: " +str(mid_doc))
  print("\n")

def out_func(movie_lines,x_axis_start):
  output,min_d, mid_doc = BMD_series(ref_doc, movie_lines, k_movie_lines)
  out_len = len(output)
  x_axis = np.array(range(x_axis_start, x_axis_start+out_len))
  prin(movie,min_d, mid_doc)
  plt.plot(x_axis, output, linestyle = "-", marker='*', label = txts_names[movie])  
  x_axis_start += out_len
  return x_axis_start
  

count = 3
i=0; t = 0; nt=0
x_axis_start = 0
k_movie_lines = 4
for movie, movie_lines in movie_dict.items():  
    if "thriller" in txts_genres[movie] and "horror" in txts_genres[movie] and "crime" in txts_genres[movie] and t<count and i%2==0:
        t +=1
        x_axis_start = out_func(movie_lines,x_axis_start)
        i += 1
    elif "thriller" not in txts_genres[movie] and "horror" not in txts_genres[movie] and "crime" not in txts_genres[movie] and nt<count and i%2!=0:
        nt +=1
        x_axis_start = out_func(movie_lines,x_axis_start)
        i += 1


plt.xlabel("Segments of k="+str(k_movie_lines) + " movie lines")
plt.ylabel("EMD between reference doc and movie segments")
plt.title("referece: Thriller Description")
#plt.legend()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -.3),
          ncol=3, fancybox=True, shadow=True)
plt.grid(True)
plt.show()

