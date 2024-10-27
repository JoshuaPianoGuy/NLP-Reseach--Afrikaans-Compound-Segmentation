# NLP-Reseach--Afrikaans-Compound-Segmentation
Code for semester long research project that looked at using pre-trained language models for Afrikaans compound segmentation

To use this repo, you first need to download the freely available dataset from the Aucopro project at: https://repo.sadilar.org/handle/20.500.12185/395

Next up, remove the duplicates, and run the split_new.py script which groups the data according to stems, removes any morpheme boundaries, replaces compound boundaries with '@' signs and randomly distributes the data into a traing partition of 80% of the total size, and validation and testing sets which are each 10% of the total size. To finish pre-processing, run the remove_empty.py script to remove empty lines from stem groupings.

Finally run the fine_tune_val.py script to fine-tune your chosen model. The model with the best f1 score on the validation set is loaded at the end. To test the model and see its predictions, run the improved_testing.py script.

This code is an adaptation of the code base used when developing LLMSegm.
M. Pranjić, M. Robnik-Šikonja, and S. Pollak, ‘LLMSegm: Surface-level Morphological Segmentation Using Large Language Model’, in Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), 2024, pp. 10665–10674.
