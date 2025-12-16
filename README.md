# Visualizing Contested Space to Optimize Defensive Outcomes After the Pass


In our Kaggle Notebook, we presented code to generate the CSD theory, but as it is computationally-intensive, only the code framework to calculate the CSDs for all given plays, and optimize.

Here we present code that if run can replicate the figures. The only things that need to be changed are the paths to align with your directory structures.
We also present 100 sample CSDs from the first 100 plays given in the NFL Dataset.

### Running the Code and File Explanation
nbdHelp.py - contains all the helper methods required
CSDTheory.ipynb - allows the generation of all the CSD theory files (identical to the file presented in Kaggle)
CSDTOTPFLegacy.ipynb - a replication of the second kaggle file for posterity. It is recommended that you do not run it.

The CSDTheory.ipynb file will allow the generation of all figures except the logistic fits, to obtain the logistic fits,

First, run COB.ipynb. COB (Change-of-basis) prepares the raw data for manipulation and stores it in a more readable format. It also decomposes the (gameid,playid) into a single (flatplayid) indexing. 
Next, run genCSD.ipynb for all desired flatplayids. Doing so will generate the CSDs.
Finally, calcTPFfunc.ipynb and pcatch.ipynb can be run to obtain the transformation from CSD to tackle probability and catch probability respectively. This file will create the logistic figures.

Generating the values in Table 1 requires optimization. First, optimize.ipynb, for the given play, will determine the possible alternative alignments for the defenders and save in a dataframe.
Then, 15analyze.ipynb will generate the expected YAC for each of those alignments (This also is a very computationally intensive process if you increase the number of alignments tested as is required). Finally, the best performing methods are identified and manually catagorized into one of the 4 cases. 
