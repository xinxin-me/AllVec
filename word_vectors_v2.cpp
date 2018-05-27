//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <vector>
#include <ctime>


#define MAX_STRING 100

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

struct vocab_word {
    int cn;
    char *word;
};


typedef struct cooccur_rec {
    int word1;
    int word2;
    double val;
} CREC;

char output_file[MAX_STRING], word_occu_file[MAX_STRING];
char read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, debug_mode = 2 /*,num_threads = 12*/;
int vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, iter = 5, classes = 0;
float lrate = 0.025, starting_lrate;
float *syn0, *syn1, *subloss;
clock_t start;
int num_threads;
long long amount=0, amount_perthread=0;
long long *label,*label_w, *label_c;
float sum;
float *w_p, *c_p;
float shift=0;
float reg=0.1;



int batch=0, amax=100;
float w0=0, alpha=0.75, thro=0.75;
float *pc, *pw, *w_neg, *qc, *qw;
std::vector<std::vector<int> > w_context_table;
std::vector<std::vector<int> > context_w_table;
std::vector<std::vector<float> > w_context_occu;
std::vector<std::vector<float> > context_w_occu;



void *UpdatePcThread(void *id) {
    long long num=(long long)id;
    int count = layer1_size/num_threads;
    float temp;
    if(num!=num_threads-1){
        for(int i=num*count;i<(num+1)*count;i++){
            for(int i1=0;i1<layer1_size;i1++){
                temp=0;
                for(int u=0;u<vocab_size;u++){
                    temp+=syn1[u*layer1_size+i]*syn1[u*layer1_size+i1]*w_neg[u];
                    //temp2+=syn0[u*layer1_size+i]*syn0[u*layer1_size+i1];
                }
                pc[i*layer1_size+i1]=temp;
                //tp[i*layer1_size+i1]=temp2;
            }
        }
    }
    else{
        for(int i=num*count;i<layer1_size;i++){
            for(int i1=0;i1<layer1_size;i1++){
                temp=0;
                for(int u=0;u<vocab_size;u++){
                    temp+=syn1[u*layer1_size+i]*syn1[u*layer1_size+i1]*w_neg[u];
                    //temp2+=syn0[u*layer1_size+i]*syn0[u*layer1_size+i1];
                }
                pc[i*layer1_size+i1]=temp;
                //tp[i*layer1_size+i1]=temp2;
            }
        }
    }
    pthread_exit(NULL);
}


void *UpdateQcThread(void *id) {
    long long num=(long long)id;
    int count = layer1_size/num_threads;
    float temp;
    if(num!=num_threads-1){
        for(int i=num*count;i<(num+1)*count;i++){
            temp=0;
            for(int u=0;u<vocab_size;u++){
                temp+=syn1[u*layer1_size+i]*w_neg[u];
                //temp2+=syn0[u*layer1_size+i]*syn0[u*layer1_size+i1];
            }
            qc[i]=temp;
            //tp[i*layer1_size+i1]=temp2;
        }
    }
    else{
        for(int i=num*count;i<layer1_size;i++){
            temp=0;
            for(int u=0;u<vocab_size;u++){
                temp+=syn1[u*layer1_size+i]*w_neg[u];
                //temp2+=syn0[u*layer1_size+i]*syn0[u*layer1_size+i1];
            }
            qc[i]=temp;
            //tp[i*layer1_size+i1]=temp2;
        }
    }
    pthread_exit(NULL);
}

void *UpdatePwThread(void *id) {
    long long num=(long long)id;
    int count = layer1_size/num_threads;
    float temp;
    if(num!=num_threads-1){
        for(int i=num*count;i<(num+1)*count;i++){
            for(int i1=0;i1<layer1_size;i1++){
                temp=0;
                for(int u=0;u<vocab_size;u++){
                    //temp+=syn1[u*layer1_size+i]*syn1[u*layer1_size+i1]*w_neg[u];
                    temp+=syn0[u*layer1_size+i]*syn0[u*layer1_size+i1];
                }
                pw[i*layer1_size+i1]=temp;
                //tp[i*layer1_size+i1]=temp2;
            }
        }
    }
    else{
        for(int i=num*count;i<layer1_size;i++){
            for(int i1=0;i1<layer1_size;i1++){
                temp=0;
                for(int u=0;u<vocab_size;u++){
                    //temp+=syn1[u*layer1_size+i]*syn1[u*layer1_size+i1]*w_neg[u];
                    temp+=syn0[u*layer1_size+i]*syn0[u*layer1_size+i1];
                }
                pw[i*layer1_size+i1]=temp;
                //tp[i*layer1_size+i1]=temp2;
            }
        }
    }
    pthread_exit(NULL);
}


void *UpdateQwThread(void *id) {
    long long num=(long long)id;
    int count = layer1_size/num_threads;
    float temp;
    if(num!=num_threads-1){
        for(int i=num*count;i<(num+1)*count;i++){
            temp=0;
            for(int u=0;u<vocab_size;u++){
                //temp+=syn1[u*layer1_size+i]*w_neg[u];
                temp+=syn0[u*layer1_size+i];
            }
            qw[i]=temp;
            //tp[i*layer1_size+i1]=temp2;
        }
    }
    else{
        for(int i=num*count;i<layer1_size;i++){
            temp=0;
            for(int u=0;u<vocab_size;u++){
                //temp+=syn1[u*layer1_size+i]*w_neg[u];
                temp+=syn0[u*layer1_size+i];
            }
            qw[i]=temp;
            //tp[i*layer1_size+i1]=temp2;
        }
    }
    pthread_exit(NULL);
}

void *UpdateParaWThread(void *id) {
    float *rui, *rui1, *weight, *pred;
    float newgrad;
    long long num=(long long)id;
    int context;
    float gradient,ga,gb,gc,gradient2;
    float occu;
    for (int w=label_w[num];w<label_w[num+1];w++){
        rui=(float *)malloc(w_context_table[w].size()*sizeof(float));
        rui1=(float *)malloc(w_context_table[w].size()*sizeof(float));
        weight=(float *)malloc(w_context_table[w].size()*sizeof(float));
        pred=(float *)malloc(w_context_table[w].size()*sizeof(float));
        for(int c=0;c<w_context_table[w].size();c++){
            context=w_context_table[w][c];
            rui[c]=0;
            for(int i=0;i<layer1_size;i++) rui[c]+=syn0[w*layer1_size+i]*syn1[context*layer1_size+i];
            occu=w_context_occu[w][c];
            if(occu>amax)
            weight[c]=1;
            else
            weight[c]=pow(occu/amax,alpha);
            pred[c]=log((occu*sum)/(w_p[w]*c_p[context]));
            if(pred[c]<0)
            pred[c]=0;
        }
        for (int i = 0; i < layer1_size; i++){
            ga=0;gb=0;gc=0;gradient=0;
            for(int i1=0;i1<layer1_size;i1++)
            ga+=(syn0[w*layer1_size+i1]*pc[i*layer1_size+i1]);
            gradient2=pc[i*layer1_size+i];
            for(int c=0;c<w_context_table[w].size();c++){
                context=w_context_table[w][c];
                rui1[c]=rui[c]-syn0[w*layer1_size+i]*syn1[context*layer1_size+i];
                gb+=(weight[c]-w_neg[context])*((weight[c]*pred[c]-w_neg[context]*shift)/(weight[c]-w_neg[context])-rui[c])*syn1[context*layer1_size+i];
                gradient2+=(weight[c]-w_neg[context])*syn1[context*layer1_size+i]*syn1[context*layer1_size+i];
            }
            gc=shift*qc[i];
            gradient=ga-gb-gc+reg*syn0[w*layer1_size+i];
            gradient2+=reg;
            syn0[w*layer1_size+i]-=gradient/gradient2;
            for(int c=0;c<w_context_table[w].size();c++){
                context=w_context_table[w][c];
                rui[c]=rui1[c]+syn0[w*layer1_size+i]*syn1[context*layer1_size+i];
            }
        }
        free(rui);
        free(rui1);
        free(weight);
        free(pred);
    }
    pthread_exit(NULL);
}

void *UpdateParaCThread(void *id) {
    float *rui, *rui1, *weight, *pred;
    float newgrad;
    long long num=(long long)id;
    int word;
    float gradient,ga,gb,gc,gradient2;
    float occu;
    
    for(int c=label_c[num];c<label_c[num+1];c++){
        rui=(float *)malloc(context_w_table[c].size()*sizeof(float));
        rui1=(float *)malloc(context_w_table[c].size()*sizeof(float));
        weight=(float *)malloc(context_w_table[c].size()*sizeof(float));
        pred=(float *)malloc(context_w_table[c].size()*sizeof(float));
        for(int j=0;j<context_w_table[c].size();j++){
            rui[j]=0;
            word=context_w_table[c][j];
            for(int i=0;i<layer1_size;i++) rui[j]+=syn0[word*layer1_size+i]*syn1[c*layer1_size+i];
            occu=context_w_occu[c][j];
            if(occu>amax)
            weight[j]=1;
            else
            weight[j]=pow(occu/amax,alpha);
            pred[j]=log((occu*sum)/(w_p[word]*c_p[c]));
            if(pred[j]<0)
            pred[j]=0;
        }
        for(int i=0;i<layer1_size;i++){
            gradient=0;ga=0;gb=0;gc=0;
            for(int i1=0;i1<layer1_size;i1++)
            ga+=(syn1[c*layer1_size+i1]*pw[i*layer1_size+i1]);
            gradient2=w_neg[c]*pw[i*layer1_size+i];
            for(int j=0;j<context_w_table[c].size();j++){
                word=context_w_table[c][j];
                rui1[j]=rui[j]-syn0[word*layer1_size+i]*syn1[c*layer1_size+i];
                gb+=(weight[j]-w_neg[c])*((weight[j]*pred[j]-w_neg[c]*shift)/(weight[j]-w_neg[c])-rui[j])*syn0[word*layer1_size+i];
                gradient2+=(weight[j]-w_neg[c])*syn0[word*layer1_size+i]*syn0[word*layer1_size+i];
            }
            gc=shift*qw[i]*w_neg[c];
            gradient=w_neg[c]*ga-gb-gc+reg*syn1[c*layer1_size+i];
            gradient2+=reg;
            syn1[c*layer1_size+i]-=gradient/gradient2;
            for(int j=0;j<context_w_table[c].size();j++){
                word=context_w_table[c][j];
                rui[j]=rui1[j]+syn0[word*layer1_size+i]*syn1[c*layer1_size+i];
            }
        }
        free(rui);
        free(rui1);
        free(pred);
        free(weight);
    }
    pthread_exit(NULL);
}

void *UpdateLossThread(void *id) {
    long long num=(long long)id;
    float occu,weight, pred;
    subloss[num]=0; float *rui; int context;
    int count= layer1_size/num_threads;
    if(num!=num_threads-1)
    for(int i=num*count;i<(num+1)*count;i++){
        for(int i1=0;i1<layer1_size;i1++)
        subloss[num]+=pw[i*layer1_size+i1]*pc[i*layer1_size+i1]*0.5;
        subloss[num]-=shift*qw[i]*qc[i];
    }
    else
    for(int i=num*count;i<layer1_size;i++){
        for(int i1=0;i1<layer1_size;i1++)
        subloss[num]+=pw[i*layer1_size+i1]*pc[i*layer1_size+i1]*0.5;
        subloss[num]-=shift*qw[i]*qc[i];
    }
    for(int w=label_w[num];w<label_w[num+1];w++){
        rui=(float *)malloc(w_context_table[w].size()*sizeof(float));
        for(int c=0;c<w_context_table[w].size();c++){
            context=w_context_table[w][c];
            occu=w_context_occu[w][c];
            if(occu>amax)
            weight=1;
            else
            weight=pow(occu/amax,alpha);
            pred=log((occu*sum)/(w_p[w]*c_p[context]));
            if(pred<0)
            pred=0;
            rui[c]=0;
            for(int i=0;i<layer1_size;i++) rui[c]+=syn0[w*layer1_size+i]*syn1[context*layer1_size+i];
            subloss[num]+=0.5*weight*(pred-rui[c])*(pred-rui[c]);
            subloss[num]-=0.5*w_neg[context]*(rui[c]-shift)*(rui[c]-shift);
        }
        free(rui);
    }
    pthread_exit(NULL);
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
    int a = 0, ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13) continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (a > 0) {
                if (ch == '\n') ungetc(ch, fin);
                break;
            }
            if (ch == '\n') {
                strcpy(word, (char *)"</s>");
                return;
            } else continue;
        }
        word[a] = ch;
        a++;
        if (a >= MAX_STRING - 1) a--;   // Truncate too long words
    }
    word[a] = 0;
}



// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
    unsigned int hash, length = strlen(word) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 0;
    vocab_size++;
    // Reallocate memory if needed
    if (vocab_size + 2 >= vocab_max_size) {
        vocab_max_size += 1000;
        vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    return vocab_size - 1;
}


void ReadVocab() {
    long long a, i = 0;
    char c;
    char word[MAX_STRING];
    FILE *fin = fopen(read_vocab_file, "rb");
    if (fin == NULL) {
        printf("Vocabulary file not found\n");
        exit(1);
    }
    vocab_size = 0;
    while (1) {
        ReadWord(word, fin);
        if (feof(fin)) break;
        a = AddWordToVocab(word);
        fscanf(fin, "%d%c", &vocab[a].cn, &c);
        i++;
    }
    for(a=0;a<vocab_size;a++)
    train_words+=vocab[a].cn;
    if (debug_mode > 0) {
        printf("Vocab size: %d\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
}

void InitNet() {
    long long a, b;
    unsigned long long next_random = 1;
    a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(float));
    if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (float)65536) - 0.5) / layer1_size;
    }
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(float));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++){
        next_random = next_random * (unsigned long long)25214903917 + 11;
        syn1[a * layer1_size + b] =(((next_random & 0xFFFF) / (float)65536) - 0.5) / layer1_size;
    }
}

/*void GetOccuMatrix(){
 int word, context;
 float occu;
 FILE *fin = fopen(word_occu_file, "rb");
 if (fin == NULL) {
 printf("Occurance file not found\n");
 exit(1);
 }
 while (1) {
 if (feof(fin)) break;
 fscanf(fin, "%d", &word);
 fscanf(fin, "%d", &context);
 fscanf(fin, "%f", &occu);
 w_context_table[word].push_back(context);
 w_context_occu[word].push_back(occu);
 context_w_table[context].push_back(word);
 context_w_occu[context].push_back(occu);
 amount++;
 }
 fclose(fin);
 }*/

void GetOccuMatrix(){
    int word, context;
    CREC cr;
    float occu;
    FILE *fin = fopen(word_occu_file, "rb");
    if (fin == NULL) {
        printf("Occurance file not found\n");
        exit(1);
    }
    while (1) {
        fread(&cr, sizeof(CREC), 1, fin);
        if (feof(fin)) break;
        word=cr.word1-1;
        context=cr.word2-1;
        occu=cr.val;
        w_context_table[word].push_back(context);
        w_context_occu[word].push_back(occu);
        context_w_table[context].push_back(word);
        context_w_occu[context].push_back(occu);
        amount++;
    }
    fclose(fin);
}

void savepara(int iter){
    char str[25];
    snprintf(str,25,"%d",iter);
    FILE* fo = fopen(str, "wb");
    // Save the word vectors
    fprintf(fo, "%d %d\n", vocab_size, layer1_size);
    float sum;
    for (int a = 0; a < vocab_size; a++) {
        fprintf(fo, "%s ", vocab[a].word);
        if (binary){
            for (int b = 0; b < layer1_size; b++){
                sum=syn0[a*layer1_size+b]+syn1[a*layer1_size+b];
                fwrite(&sum, sizeof(float), 1, fo);
            }
            fflush(fo);
        }
        else{
            for (int b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]+syn1[a*layer1_size+b]);
            fflush(fo);
        }
        fprintf(fo, "\n");
    }
}


void TrainModel() {
    long a,b,c,d;
    FILE *fo;
    printf("Starting training using file %s\n", word_occu_file);
    starting_lrate = lrate;
    ReadVocab();
    if (output_file[0] == 0) return;
    InitNet();
    start = clock();
    pc=(float *)malloc(layer1_size*layer1_size*sizeof(float));
    pw=(float *)malloc(layer1_size*layer1_size*sizeof(float));
    qc=(float *)malloc(layer1_size*sizeof(float));
    qw=(float *)malloc(layer1_size*sizeof(float));
    w_neg= (float *)malloc(vocab_size* sizeof(float));
    subloss=(float *)malloc(num_threads* sizeof(float));
    label_w=(long long *)malloc(sizeof(long long)*(num_threads+1)); label_w[0]=0; label_w[num_threads]=vocab_size-1;
    label_c=(long long *)malloc(sizeof(long long)*(num_threads+1)); label_c[0]=0; label_c[num_threads]=vocab_size-1;
    label=(long long *)malloc(sizeof(long long)*(num_threads-1));
    for(int i=0;i<vocab_size;i++) w_context_table.push_back(*new std::vector<int>);
    for(int i=0;i<vocab_size;i++) context_w_table.push_back(*new std::vector<int>);
    for(int i=0;i<vocab_size;i++) w_context_occu.push_back(*new std::vector<float>);
    for(int i=0;i<vocab_size;i++) context_w_occu.push_back(*new std::vector<float>);
    
    pthread_t *updatepw= (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    pthread_t *updatepc= (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    pthread_t *updateqw= (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    pthread_t *updateqc= (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    pthread_t *updateparaw= (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    pthread_t *updateparac= (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    pthread_t *updateloss=(pthread_t *)malloc(num_threads * sizeof(pthread_t));
    
    GetOccuMatrix();
    printf("Get word occurance matrix completed\n");
    amount_perthread=amount/num_threads;
    for(int i=0;i<num_threads-1;i++) label[i]=amount_perthread*(i+1);
    
    //for(int i=0;i<num_threads-1;i++) printf("%d\n",label[i]);
    
    for(int i=0;i<num_threads-1;i++){
        long long count=0;
        for(int j=0;j<vocab_size;j++){
            count+=w_context_table[j].size();
            if(count>label[i]){
                label_w[i+1]=j;
                break;
            }
        }
    }
    
    
    //for(int i=0;i<num_threads-1;i++) printf("%d\n",label[i]);
    
    for(int i=0;i<num_threads-1;i++){
        long long count=0;
        for(int j=0;j<vocab_size;j++){
            count+=context_w_table[j].size();
            if(count>label[i]){
                label_c[i+1]=j;
                break;
            }
        }
    }
    
    sum=0;
    float Z=0;
    float *p=(float *)malloc(vocab_size* sizeof(float));
    for(int i=0;i<vocab_size;i++) p[i]=0;
    for(int i=0;i<vocab_size;i++){
        for(int j=0;j<context_w_table[i].size();j++)
        p[i]+=context_w_occu[i][j];
        sum+=p[i];
    }
    
    for(int i=0;i<vocab_size;i++){
        p[i]/=sum;
        p[i]=pow(p[i], thro);
        Z+=p[i];
    }
    
    for(int i=0;i<vocab_size;i++)
    w_neg[i]=w0*p[i]/Z;
    free(p);
    printf("Update w_neg completed\n");
    
    w_p=(float *)malloc(vocab_size* sizeof(float));
    c_p=(float *)malloc(vocab_size* sizeof(float));
    for(int i=0;i<vocab_size;i++){w_p[i]=0; c_p[i]=0;}
    for(int i=0;i<vocab_size;i++){
        for(int j=0;j<context_w_table[i].size();j++)
        c_p[i]+=context_w_occu[i][j];
        for(int j=0;j<w_context_table[i].size();j++)
        w_p[i]+=w_context_occu[i][j];
    }
    printf("Start training parameters\n");
    
    for(int k=0;k<iter;k++){
        for (a = 0; a < num_threads; a++) pthread_create(&updatepc[a], NULL, UpdatePcThread, (void *)a);
        for (a = 0; a < num_threads; a++) pthread_join(updatepc[a], NULL);
        for (a = 0; a < num_threads; a++) pthread_create(&updateqc[a], NULL, UpdateQcThread, (void *)a);
        for (a = 0; a < num_threads; a++) pthread_join(updateqc[a], NULL);
        for (a = 0; a < num_threads; a++) pthread_create(&updateparaw[a], NULL, UpdateParaWThread , (void *)a);
        for (a = 0; a < num_threads; a++) pthread_join(updateparaw[a], NULL);
        
        for (a = 0; a < num_threads; a++) pthread_create(&updatepw[a], NULL, UpdatePwThread, (void *)a);
        for (a = 0; a < num_threads; a++) pthread_join(updatepw[a], NULL);
        for (a = 0; a < num_threads; a++) pthread_create(&updateqw[a], NULL, UpdateQwThread, (void *)a);
        for (a = 0; a < num_threads; a++) pthread_join(updateqw[a], NULL);
        for (a = 0; a < num_threads; a++) pthread_create(&updateparac[a], NULL, UpdateParaCThread , (void *)a);
        for (a = 0; a < num_threads; a++) pthread_join(updateparac[a], NULL);
        
        for (a = 0; a < num_threads; a++) pthread_create(&updatepc[a], NULL, UpdatePcThread, (void *)a);
        for (a = 0; a < num_threads; a++) pthread_join(updatepc[a], NULL);
        for (a = 0; a < num_threads; a++) pthread_create(&updateqc[a], NULL, UpdateQcThread, (void *)a);
        for (a = 0; a < num_threads; a++) pthread_join(updateqc[a], NULL);
        for (a = 0; a < num_threads; a++) pthread_create(&updateloss[a], NULL, UpdateLossThread, (void *)a);
        for (a = 0; a < num_threads; a++) pthread_join(updateloss[a], NULL);
        float loss=0;
        for(int i=0;i<num_threads;i++) loss+=subloss[i];
        loss+=shift*shift*vocab_size*w0;
        printf("the total loss in %dth iter is %f\n",k,loss);
    }
    free(qw);
    free(qc);
    free(pw);
    free(pc);
    free(w_neg);
    free(w_p);
    free(c_p);
    fo = fopen(output_file, "wb");
    if (classes == 0) {
        // Save the word vectors
        fprintf(fo, "%d %d\n", vocab_size, layer1_size);
        float sum;
        for (a = 0; a < vocab_size; a++) {
            fprintf(fo, "%s ", vocab[a].word);
            if (binary){
                for (b = 0; b < layer1_size; b++){
                    sum=syn0[a*layer1_size+b]+syn1[a*layer1_size+b];
                    fwrite(&sum, sizeof(float), 1, fo);
                }
                fflush(fo);
            }
            else{
                for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]+syn1[a*layer1_size+b]);
                fflush(fo);
            }
            fprintf(fo, "\n");
        }
    } else {
        // Run K-means on the word vectors
        int clcn = classes, iter = 10, closeid;
        int *centcn = (int *)malloc(classes * sizeof(int));
        int *cl = (int *)calloc(vocab_size, sizeof(int));
        float closev, x;
        float *cent = (float *)calloc(classes * layer1_size, sizeof(float));
        for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
        for (a = 0; a < iter; a++) {
            for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
            for (b = 0; b < clcn; b++) centcn[b] = 1;
            for (c = 0; c < vocab_size; c++) {
                for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
                centcn[cl[c]]++;
            }
            for (b = 0; b < clcn; b++) {
                closev = 0;
                for (c = 0; c < layer1_size; c++) {
                    cent[layer1_size * b + c] /= centcn[b];
                    closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
                }
                closev = sqrt(closev);
                for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
            }
            for (c = 0; c < vocab_size; c++) {
                closev = -10;
                closeid = 0;
                for (d = 0; d < clcn; d++) {
                    x = 0;
                    for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
                    if (x > closev) {
                        closev = x;
                        closeid = d;
                    }
                }
                cl[c] = closeid;
            }
        }
        // Save the K-means classes
        for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
        free(centcn);
        free(cent);
        free(cl);
    }
    fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("WORD VECTOR estimation toolkit v 0.1c\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-train <file>\n");
        printf("\t\tUse text data from <file> to train the model\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of word vectors; default is 100\n");
        printf("\t-window <int>\n");
        printf("\t\tSet max skip length between words; default is 5\n");
        printf("\t-sample <float>\n");
        printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
        printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
        printf("\t-hs <int>\n");
        printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 12)\n");
        printf("\t-iter <int>\n");
        printf("\t\tRun more training iterations (default 5)\n");
        printf("\t-min-count <int>\n");
        printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
        printf("\t-lrate <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
        printf("\t-classes <int>\n");
        printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
        printf("\t-debug <int>\n");
        printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
        printf("\t-save-vocab <file>\n");
        printf("\t\tThe vocabulary will be saved to <file>\n");
        printf("\t-read-vocab <file>\n");
        printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
        printf("\t-cbow <int>\n");
        printf("\t\tUse the continuous bag of words model; default is 1 (use 0 for skip-gram model)\n");
        printf("\nExamples:\n");
        printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1 -iter 3\n\n");
        return 0;
    }
    output_file[0] = 0;
    read_vocab_file[0] = 0;
    word_occu_file[0]=0;
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-word-occu", argc, argv)) > 0) strcpy(word_occu_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
    //if (cbow) lrate = 0.05;
    if ((i = ArgPos((char *)"-lrate", argc, argv)) > 0) lrate = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-iter", argc, argv)) > 0) iter = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-batch", argc, argv)) > 0) batch = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-amax", argc, argv)) > 0) amax = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-w0", argc, argv)) > 0) w0 = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-thro", argc, argv)) > 0) thro = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-shift", argc, argv)) > 0) shift = atof(argv[i + 1]);
    vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    TrainModel();
    return 0;
}





