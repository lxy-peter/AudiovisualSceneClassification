import json
import os





def read_classes():

    # Writing class ID to class name file
    f = open('data/orig/ontology.json', 'r')
    ontology = json.load(f)
    f.close()

    classid_to_classname = {}
    for label in ontology:
        label_name = label['name'].lower().strip()
        classid_to_classname[label['id']] = label_name

    f = open('data/orig/classid_to_classname.txt', 'w')
    for classid in classid_to_classname:
        f.write(classid + ' ' + classid_to_classname[classid] + '\n')
    f.close()


    # Writing class ID to class index file
    classids = list(classid_to_classname.keys())
    classid_to_classidx = {}
    f = open('data/orig/classidx_to_classid.txt', 'w')
    for i, classid in enumerate(classids):
        f.write(str(i) + ' ' + classid + '\n')
        classid_to_classidx[classid] = i
    f.close()


    f = open('audiosetdl/dataset/eval_segments.csv', 'r')
    csv = [line for line in f.read().split('\n') if not line.startswith('#')]
    try:
        while True:
            csv.remove('')
    except ValueError:
        pass


    # Writing sample ID to class ID file, and
    # sample ID to class index file
    labels_classid = {}
    labels_classidx = {}
    for line in csv:
        pos = line.find('"')
        sampleid = line[:pos].split(',')[0]
        labs = line[pos:].replace('"', '').split(',')
        labs = [lab.strip() for lab in labs]
        
        labels_classid[sampleid] = labs
        labels_classidx[sampleid] = [classid_to_classidx[lab] for lab in labs]


    f1 = open('data/orig/sampleid_to_classid.txt', 'w')
    f2 = open('data/orig/sampleid_to_classidx.txt', 'w')
    for sampleid in labels_classid:
        classids = labels_classid[sampleid]
        classidxs = labels_classidx[sampleid]
        
        line_f1 = sampleid + ' '
        line_f2 = sampleid + ' '
        
        for label in classids:
            line_f1 += label + ' '
        line_f1 = line_f1.strip()
        
        for label in classidxs:
            line_f2 += str(label) + ' '
        line_f2 = line_f2.strip()
        
        f1.write(line_f1 + '\n')
        f2.write(line_f2 + '\n')
        
    f1.close()
    f2.close()



def count_classes():
    
    f = open('data/orig/sampleid_to_classid.txt', 'r')
    lines = [line for line in f.read().split('\n') if line != '']
    f.close()
    
    sampleid_to_classid = {}
    for line in lines:
        first_space_pos = line.find(' ')
        sampleid = line[:first_space_pos]
        classids = line[first_space_pos+1:].split(' ')
        sampleid_to_classid[sampleid] = classids
    
    
    f = open('data/orig/classid_to_classname.txt', 'r')
    lines = [line for line in f.read().split('\n') if line != '']
    f.close()
    
    classid_to_classname = {}
    for line in lines:
        first_space_pos = line.find(' ')
        classid = line[:first_space_pos]
        classnames = line[first_space_pos+1:]
        classid_to_classname[classid] = classnames
    
    
    audio_wavs = [file for file in os.listdir('audiosetdl/dataset/data/eval_segments/audio_wav') if not file.startswith('.')]
    relevant = {}
    for file in audio_wavs:
        pos = file.rfind('_')
        sampleid = file[:pos]
        pos = sampleid.rfind('_')
        sampleid = sampleid[:pos]
        relevant[sampleid] = sampleid_to_classid[sampleid]
    
    # classname_counts = {}
    classid_counts = {}
    for sampleid in relevant:
        classids = relevant[sampleid]
        for classid in classids:
            # classname = classid_to_classname[classid]
            if classid in classid_counts:
                classid_counts[classid] += 1
            else:
                classid_counts[classid] = 1
                
    classid_counts = sorted(classid_counts.items(), key=lambda x: x[1])
    f = open('audiosetdl/dataset/data/eval_segments/classid_counts.txt', 'w')
    for item in classid_counts:
        f.write(item[0] + ' ' + str(item[1]) + '\n')
    f.close()
    
    top_classes = classid_counts[-12:-2] # do not include generic classes speech and music
    f = open('audiosetdl/dataset/data/eval_segments/top_classes.txt', 'w')
    for item in top_classes:
        f.write(item[0] + ' ' + str(item[1]) + '\n')
    f.close()



def revise_classes():
    
    f = open('audiosetdl/dataset/data/eval_segments/top_classes.txt', 'r')
    keep = [] # will contain class IDs
    for line in f.read().split('\n'):
        if line != '':
            pos = line.find(' ')
            keep.append(line[:pos])
    f.close()
    
    # Read in data/orig/classid_to_classname.txt
    f = open('data/orig/classid_to_classname.txt', 'r')
    lines = [line for line in f.read().split('\n') if line != '']
    f.close()
    classid_to_classname = {}
    for line in lines:
        pos = line.find(' ')
        classid = line[:pos]
        if classid in keep:
            classname = line[pos+1:]
            classid_to_classname[classid] = classname
    
    # Write new classid_to_classname.txt
    f = open('data/classid_to_classname.txt', 'w')
    for classid in classid_to_classname:
        f.write(classid + ' ' + classid_to_classname[classid] + '\n')
    f.close()
    
    # Write new data/classidx_to_classid.txt
    f = open('data/classidx_to_classid.txt', 'w')
    classidx_to_classid = {}
    classid_to_classidx = {}
    for i, classid in enumerate(classid_to_classname):
        classidx_to_classid[i] = classid
        classid_to_classidx[classid] = i
        f.write(str(i) + ' ' + classid + '\n')
    f.close()
    
    # Read in data/orig/sampleid_to_classid.txt
    f = open('data/orig/sampleid_to_classid.txt', 'r')
    lines = [line for line in f.read().split('\n') if line != '']
    f.close()
    
    # Write new data/sampleid_to_classid.txt
    f1 = open('data/sampleid_to_classid.txt', 'w')
    f2 = open('data/sampleid_to_classidx.txt', 'w')
    sampleid_to_classid = {}
    for line in lines:
        pos = line.find(' ')
        sampleid = line[:pos]
        classes = line[pos+1:].split(' ')
        classes = set(classes).intersection(set(keep))
        
        if len(classes) == 0:
            continue
        
        sampleid_to_classid[sampleid] = list(classes)
        
        f1.write(sampleid)
        f2.write(sampleid)
        for classid in classes:
            f1.write(' ' + classid)
            f2.write(' ' + str(classid_to_classidx[classid]))
        f1.write('\n')
        f2.write('\n')
    f1.close()
    f2.close()
    
    for split in ['train', 'dev', 'test']:
        f_read = open('audiosetdl/dataset/data/eval_segments/orig/' + split + '.txt', 'r')
        f_write = open('audiosetdl/dataset/data/eval_segments/' + split + '.txt', 'w')
        
        for line in f_read.read().split('\n'):
            if line != '':
                pos = line.rfind('_')
                sampleid = line[:pos]
                pos = sampleid.rfind('_')
                sampleid = sampleid[:pos]
                
                if sampleid in sampleid_to_classid:
                    f_write.write(sampleid + '\n')
        
        f_read.close()
        f_write.close()
                
                


if __name__ == '__main__':
    read_classes()
    count_classes()
    revise_classes()
    
    
    
    
    
