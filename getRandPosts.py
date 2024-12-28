import json
import random

def extract_random_posts(input_file, output_file, num_posts=200, num_comments=5):
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    
    selected_posts = random.sample(data, min(len(data), num_posts))
    
    for post in selected_posts:
        if 'comments' in post:
            post['comments'] = random.sample(post['comments'], min(len(post['comments']), num_comments))
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(selected_posts, outfile, indent=4, ensure_ascii=False)

input_file = 'multi_few_data.json' 
output_file = 'labeledSet.json' 

extract_random_posts(input_file, output_file)