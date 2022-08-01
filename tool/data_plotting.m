%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Project: Facial Emotion Recognition using CNN in PyTorch
% Author: Deyuan Qu, Sudip Dhakal, Dominic Carrillo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
clear;
clc;

% Read file from local folder
train_data = load('C:\Users\dqu\Desktop\CV_Project\result\Test1\accuracy_train.txt');
val_data = load('C:\Users\dqu\Desktop\CV_Project\result\Test1\accuracy_val.txt');
loss_rate_data = load('C:\Users\dqu\Desktop\CV_Project\result\Test1\loss_rate.txt');

% epochs = 100
x = 1:1:100;
y = train_data;
y2 = val_data;
y3 = loss_rate_data;
% Plot accuracy of training dataset and validation dataset
figure(1)
plot(x,y,x,y2)
title('Training accuracy vs Validation accuracy')
xlabel('epochs = 100') 
ylabel('accuarcy %')
legend({'train acc','val acc'},'Location','northeast')
% Plot loss rate of training dataset
figure(2)
plot(x,y3)
title('Loss rate of training dataset')
xlabel('epochs = 100') 
ylabel('percentage %')