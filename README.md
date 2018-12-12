# M2R
This work is from Weston et.al (2013) *Connecting Language and Knowledge Bases with Embedding Models for Relation Extraction*.
We only implement m2r and the composite scoring.
## Training
Set option [loadFromData], [testFlag] False
```
class HyperPara:
    def __init__(self):
        ...
        # if you want to test, set these two options True
        self.loadFromData = False
        self.testFlag = False
        # use Sm2r+kb to evaluate relation extraction
        self.composite = True
```
## Testing
Set option [loadFromData] and [testFlag] True
In the testing phase, we use pretrained TransE model. The *relation2vec.bern* and *entity2vec.bern* in *data/* are the trained model
using fb40k.
## Seek for help
**UNFORTUNATELY!** My result is totally different from paper's as well as other papers'. There are maybe some reasons:
1) Since I am not good at implementing algorithm and a freshman to Tensorflow, the implementation goes wrong.
2) Dataset nyt+freebase I download (from http://iesl.cs.umass.edu/riedel/data-univSchema/, nyt-freebase.dev.universal.ds.txt as testing file
and nyt-freebase.train.universal.ds.txt as training file) may be far different from those papers used. It's quite hard to determine what 
the nyt+freebase is and where to download it.
3) I use the *sameas.org* to crawl MID of each entity in the nyt+freebase. And I filter out entities having no MID which account for
a large part. Anyway, the file after processing is in the directory Dataset.
4) **Last but very important**. I just cannot get why the recall reported in papers is so low while the precision is high. Given an entity pair (h,t),
the number of rels between h and t usually is 1. In this case, the recall should be high if the precision is high. And with the declining
of precision, the recall will continue go up. 
## updaing
Updating on 12/12/2018, After refering to the code from Hoffmann et.al (Knowledge-BasedWeak Supervision for Information Extraction of Overlapping Relations, I figure out the calculation of precision and recall. It takes all rel relations in the testing set as the answer collection. However, the results are still not good. The model suffers from lack of training data.
