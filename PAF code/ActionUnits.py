from pympi.Elan import Eaf
import pandas as pd
import numpy as np

freq=50
for i in [9,10,12,15,18,19,24,26,27,30]
    data = pd.read_csv(r"C:\Users\hp\OneDrive\Bureau\PAF\processed_"+i+"\cam-table-wall.csv")

    last_18_columns = data.iloc[:, -18:]

    vectors = last_18_columns.values.tolist()
    npvectors = np.array(vectors)
    print(len(npvectors))

    #for vector in vectors[:10]: print(vector)

    eaf=Eaf(r"PAF\PAF_2023\Dataset\Interactions\"+i+"\"+i+".eaf")
    annots=sorted(eaf.get_annotation_data_for_tier('Trust'))

    X = np.zeros((len(annots), 18))
    num_segment = 0

    for segment in annots:

        fi=int(segment[0]*(0.001)*freq)
        ff=int(segment[1]*(0.001)*freq)
        #print(fi,ff)
        for i in range(fi,ff+1):
            X[num_segment]+=npvectors[i]

        for j in range(18):
            if X[num_segment][j]>0: X[num_segment][j]=1

        saved_file=r"PAF\data_AUs\"+i+"\segment_"+str(num_segment)
        np.save(saved_file,X[num_segment])

        num_segment+=1




