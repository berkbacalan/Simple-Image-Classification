clear all;


setDir  = fullfile('DATASET PATH');
imds = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource','foldernames');
[trainingSet,testSet] = splitEachLabel(imds,0.8,'randomize');
bag = bagOfFeatures(trainingSet);
categoryClassifier = trainImageCategoryClassifier(trainingSet,bag);
confMatrix = evaluate(categoryClassifier,testSet)
mean(diag(confMatrix))
img = imread(fullfile(setDir,'boy','TARGET IMAGE'));
[labelIdx, score] = predict(categoryClassifier,img);
gender=categoryClassifier.Labels(labelIdx)
imshow(img);
title(gender);



