
NfoldCV = 10;
   
prompt1="Please enter the name of dataset (e.g., iris0.txt): ";
dataset = input(prompt1,'s');

fidExp=fopen('ExpRes_H-SMOTE-FAM_nFCV.txt','a+'); 
fprintf(fidExp,"H-SMOTE-FAM\n"); 
fprintf(fidExp,'Dataset: %s\n',dataset);
fclose(fidExp);                                                         

input_data=load(dataset);

% Stratify NFCV
input_class=input_data(:,end);
cv=random_generate_stratify(input_class);

for nf = 1:NfoldCV

    train_data   = [];
    test_data    = [];

    %Stratify NFCV
    trainingClasses=input_class(cv.training(nf));

    trainingClasses=categorical(trainingClasses);
    nCounts=countcats(trainingClasses);
    nTrainingData(nf,:)=nCounts';

    CV_training_data{nf} = input_data(cv.training(nf),:);
    train_data=CV_training_data{nf};

    testClasses=input_class(cv.test(nf));

    testClasses=categorical(testClasses);
    nCounts=countcats(testClasses);
    nTestData(nf,:)=nCounts';

    CV_test_data{nf}=input_data(cv.test(nf),:);
    test_data = CV_test_data{nf};
    
    

    c = size(train_data,2);
    trainc0=train_data(train_data(:,c)==0,:);
    trainc1=train_data(train_data(:,c)==1,:);

    [trainc1] = H_SMOTE(trainc0,trainc1);

    train_data = [trainc0;trainc1];

    n=size(train_data,1);
    randorder=random_generate_train(n);
    train_data=train_data(randorder',:);

    %Ensure the data >=0 & <=1
    train_data(train_data<0)=0;
    train_data(train_data>1)=1;

    
    %Prepare 10 fold train data for FAM
    ptrain_data=fopen('train_data.txt','a+');
    [row_train, col_train] = size(train_data);
    for i=1:row_train
        fprintf(ptrain_data, '%g\t', train_data(i,:));
        fprintf(ptrain_data, '\n');
    end
    fclose(ptrain_data);

    ptrain_data_no=fopen('train_data_no.txt','a+');
    row_train_no = size(train_data,1);
    fprintf(ptrain_data_no, '%d\n', row_train_no);
    fclose(ptrain_data_no);

    %Prepare 10 fold test data for FAM
    ptest_data=fopen('test_data.txt','a+');
    [row_test, col_test] = size(test_data);
    for i=1:row_test
        fprintf(ptest_data, '%g\t', test_data(i,:));
        fprintf(ptest_data, '\n');
    end
    fclose(ptest_data);

    ptest_data_no=fopen('test_data_no.txt','a+');
    row_test_no = size(test_data,1);
    fprintf(ptest_data_no, '%d\n', row_test_no);
    fclose(ptest_data_no);
end

system('FAM');
  
clear all;


function cv=random_generate_stratify(input_class)
cv=cvpartition(input_class,'KFold',10,'Stratify',true);
end

function [order,varargout]=random_generate_train(Total_Patterns)
order = randperm(Total_Patterns);
end

function [trainc1] = H_SMOTE(trainc0,trainc1)
    
    %Remove class label
    trainc=trainc1(:,1:end-1);
    
    %Compute position of each sample
    x=prod(trainc,2);

    max_x=max(x);
    min_x=min(x);
    
    %Initial Bin Number
    B=30;
    
    Bin=cell(B,1);
    
    %Calculate Bin Width
    gap=(max_x-min_x)/size(Bin,1);

    %Define Interval Range for each bin
    for i=1:size(Bin,1)
        Bin{i}.range=gap*i;
    end

    %Initialise data space in Bins
    for i=1:size(Bin,1)
        Bin{i}.data=[];
    end

    %Distribute samples to Bins based on their positions
    for i=1:size(x,1)
        for y=1:size(Bin,1)
            if x(i)>=0 && x(i)<=Bin{y}.range && x(i)<=1    
                Bin{y}.data=cat(1,Bin{y}.data,trainc(i,:));    
                break;
            end    
        end
    end
    
    %Define Available Bin 
    Ava_Bin=0;
    for i=1:B
        if ~isempty(Bin{i}.data)
            Ava_Bin=Ava_Bin+1;
        end
    end
    
    samplepergrp=round(size(trainc0,1)/Ava_Bin);
    
    %Define Eliminated Bin 
    Eli_Bin=0;
    sizeofexceed=0;
    id2=[];
    id1=[];
    for i=1:B
        if isempty(Bin{i}.data)
            continue;
        elseif size(Bin{i}.data,1) >= samplepergrp || size(Bin{i}.data,1) == 1
            id2=cat(1,id2,i);
            Ava_Bin=Ava_Bin-1;
            Eli_Bin=Eli_Bin+1;
            sizeofexceed=sizeofexceed+size(Bin{i}.data,1);
        else
            id1=cat(1,id1,i);
        end          
    end
    
    samplepergrp=round((size(trainc0,1)-sizeofexceed)/Ava_Bin);
    
    %Number of k Nearest Neighbours
    k=5;  
    
    %Variable to store original & synthetic samples
    Sampled1=[];
    Sampled2=[];
    
    %SMOTE
    for i=1:size(Bin,1)
        if isempty(Bin{i}.data)
            continue;
        else
            if ismember(i,id1)
                T=size(Bin{i}.data,1);
                
                %Over-sampling ratio
                N=samplepergrp/size(Bin{i}.data,1);
                N=ceil(N);

                X_smote=[];

                for a=1:T
                    y=Bin{i}.data(a,:);
                    [idx,~]=knnsearch(Bin{i}.data,y,'k',k);
                    idx=datasample(idx,N);

                    x_near=Bin{i}.data(idx,:);
                    x_syn = bsxfun(@plus,bsxfun(@times,bsxfun(@minus,x_near,y),rand(N,1)),y);
                    X_smote=cat(1,X_smote,x_syn);
                end
                
                Syn_needed=samplepergrp-size(Bin{i}.data,1);
                k=size(X_smote,1);
                randorder=randperm(k);

                Sample_syn=X_smote(randorder(1:Syn_needed),:);
                Sample_ori_syn=[Bin{i}.data; Sample_syn];

                Sampled1=cat(1,Sampled1,Sample_ori_syn);
            else
                Sampled2=cat(1,Sampled2,Bin{i}.data);
            end
        end
    end    
    
    %Assign the original class label to samples
    if ~isempty(Sampled1) && isempty(Sampled2)
        f=size(Sampled1,2);
        Sampled1(:,f+1)=1;
        trainc1=Sampled1;
    elseif ~isempty(Sampled2) && isempty(Sampled1)
        f=size(Sampled2,2);
        Sampled2(:,f+1)=1;
        trainc1=Sampled2;
    else
        f=size(Sampled1,2);
        Sampled1(:,f+1)=1;
        Sampled2(:,f+1)=1;
        trainc1=[Sampled1;Sampled2];
    end
end