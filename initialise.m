%% EGH443 Assignment 2 - Channel Delay Spread
%  Use this file to generate the data that you will be using in
%  Assignment 2.
%  You will only need to run this code ONCE.
%  A .mat file will be generated for you which includes
%  the data that you will be using in this assignment.
%  - Proceed to Line 15 of the code, and fill in the
%    student numbers of your group members.
%  - Then run the code.

% Preparing MATLAB workspace
% clear, clc, close all;
%  Enter your student numbers into lines 22 to 24.
%  Student numbers have the format "n01234567".
%  Omit the leading 'n' and leading '0', and enter it as 1234567.
%  If there are only two people in your group, take the sum of the student
%  numbers and discard the leading digit. Then insert the result into st3.
% st1 = 9154884; % insert student number here
% st2 = 8538247; % insert student number here
% st3 = 7562454; %insert student number here

st1=9630210; %student 1 
st2=9713310; %student 2 
st3=9698701; %student 3


%% STOP STOP STOP
%% Please do not chage the code below

st = [st1,st2,st3];

% Generating your data.
[tvec pvec, d, LdBShadow] = A2GenData(st);

save channel.mat tvec pvec d LdBShadow;
clc
fprintf('Your Channel.mat files has been saved to:\n%s\n', pwd);
fprintf('Use the "load" command to load the saved variables into the workspace.\n');
% You only need to run this file ONCE to generate your signals.




% clear all
