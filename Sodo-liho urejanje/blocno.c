#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include <stddef.h>

#define NTHREADS 4
#define N 21
#define __PRINT__
#define PREPREKA pthread_barrier_wait(&prepreka);

typedef struct {
    int id;
    int ostanek;
} args_t;

int* pseznam;
pthread_t nit[NTHREADS];
args_t arg[NTHREADS];
pthread_barrier_t prepreka;
bool lokalno_urejeno[NTHREADS];
bool urejeno = false;

void* uredi_seznam(void* args);
void* sodo_lihi_prehod(int* id);
void* sodo_sodi_prehod(int* id);
void* liho_sodi_prehod(int* id);
void* liho_lihi_prehod(int* id);
int doloci_st_elementov(int* id);
void* primerjaj_in_zamenjaj(int* a, int* b, bool* lokalno_urejeno, int* nit, int* j);
void izpisi_seznam(void);

int main(){

    pseznam = (int*)malloc(N*sizeof(int));
    pthread_barrier_init(&prepreka, NULL, NTHREADS);

    srand(time(NULL));
    for(int i = 0; i < N; i++){
        *(pseznam + i) = rand() % 100;
    }
#ifdef __PRINT__
    izpisi_seznam();
    printf("\n\n");
#endif    
    int ostanek = N%NTHREADS;
    for (int i = 0; i < NTHREADS; i++)
    {   
        arg[i].id = i;
        arg[i].ostanek = ostanek;
        pthread_create(
            &nit[i],
            NULL, 
            uredi_seznam,
            &arg[i]
        );
    }

    for (int i = 0; i < NTHREADS; i++)
    {
        pthread_join(nit[i], NULL);
    }
    
}

void* primerjaj_in_zamenjaj(int* a, int* b, bool* lokalno_urejeno, int* nit, int *index){
#ifdef __PRINT_DEBUG__
    printf("Thread %d: Passing j = %d to primerjaj_in_zamenjaj\n", *nit, *index);
#endif
    if(*a > *b){
        int temp = *a;
        *a = *b; 
        *b = temp;
        *lokalno_urejeno = false;
#ifdef __PRINT__
        printf("Menjava elementov %d (index %d) in %d (index %d) (Nit %d)\n", *a, (*index) + 1, *b, *index, *nit);
#endif
    } else {
#ifdef __PRINT__DEBUG__
        printf("Primerjava elementov %d (index %d) in %d (index %d) (Nit %d)\n", *a, (*index) + 1, *b, *index, *nit);
#endif
    }
}

void* uredi_seznam(void* arg){
    args_t* pargumenti = (args_t*) arg;
    int id = pargumenti->id;
    int mode = pargumenti->ostanek;
    int counter = 0;

    while(!urejeno){
        lokalno_urejeno[id] = true;
        if(mode == 0){
            // ---------- SODI PREHOD ----------
            sodo_sodi_prehod(&id);
            PREPREKA
#ifdef __PRINT__
            if(id == 0){
                printf("   SODI: ");
                izpisi_seznam();
            }
#endif
            PREPREKA
            // ---------- LIHI PREHOD ----------
            sodo_lihi_prehod(&id);
            PREPREKA
#ifdef __PRINT__
            if(id == 0){
                printf("   LIHI: ");
                izpisi_seznam();
                printf("   -------------- ITERACIJA %d ----------------\n", counter);
            }
#endif
        } else {
            // ---------- SODI PREHOD ----------
            liho_sodi_prehod(&id);
            PREPREKA
#ifdef __PRINT__
            if(id == 0){
                printf("   SODI: ");
                izpisi_seznam();
            }
#endif
            PREPREKA
            // ---------- LIHI PREHOD ----------
            liho_lihi_prehod(&id);
            PREPREKA
#ifdef __PRINT__
            if(id == 0){
                printf("   LIHI: ");
                izpisi_seznam();
                printf("   -------------- ITERACIJA %d ----------------\n", counter);
            }
#endif
        }

        PREPREKA
        if(id == 0){
            bool temp = true;
            for (int i = 0; i < NTHREADS; i++)
            {
                if(!lokalno_urejeno[i]){
                    temp = false;
                    break;
                }
            }
            if(temp){
                urejeno = true;
            }
            
        }
        PREPREKA
        counter++;
    }
}

void* sodo_sodi_prehod(int* id){
    int st_elementov = doloci_st_elementov(id);

    // Sodi obhod.
    for (int i = 0; i < (st_elementov); i+=2)
    {
        int j = (*id * st_elementov) + i;
        if((j + 1) < N){
            primerjaj_in_zamenjaj(&pseznam[j], &pseznam[j+1], &lokalno_urejeno[*id], id, &j);
        }    
    }

    return NULL;
}

void* sodo_lihi_prehod(int* id) {
    int st_elementov = doloci_st_elementov(id);


    for (int i = 1; i < st_elementov; i += 2) {
        int j = (*id * (st_elementov)) + i;
        if ((j + 1) < N) {
            primerjaj_in_zamenjaj(&pseznam[j], &pseznam[j + 1], &lokalno_urejeno[*id], id, &j);
        }
    }

    return NULL;
}

void* liho_sodi_prehod(int* id){
    int st_elementov = doloci_st_elementov(id);
    if((*id % 2) == 0){  // Nit je soda
        // printf("SEM SODA NIT %d, I: %d, ST_ELEMENTOV: %d\n", *id, (N/NTHREADS) * (*id), st_elementov);
        for (int i = 0; i < st_elementov; i+=2)
        {   
            int j = ((N/NTHREADS) * (*id)) + i;
            if((j + 1) <= ((st_elementov * ((*id) + 1)) + 1) && (j + 1) < N){
                primerjaj_in_zamenjaj(&pseznam[j], &pseznam[j+1], &lokalno_urejeno[*id], id, &j);
            }
        }
    } else {  //Nit je liha
        // printf("SEM LIHA NIT %d, I: %d, ST_ELEMENTOV: %d\n", *id, ((N/NTHREADS) * (*id)) + 1, st_elementov);
        for (int i = 1; i < (st_elementov); i+=2)
        {   
            int j = ((N/NTHREADS) * (*id)) + i;
            if((i + 1) <= (st_elementov * ((*id) + 1)) && (j + 1) < N){
                primerjaj_in_zamenjaj(&pseznam[j], &pseznam[j+1], &lokalno_urejeno[*id], id, &j);
            }
        }
    }
    return NULL;
}

void* liho_lihi_prehod(int* id){
    int st_elementov = doloci_st_elementov(id);
    if((*id % 2) == 0){  // Nit je soda
        // printf("SEM SODA NIT %d, I: %d, ST_ELEMENTOV: %d\n", *id, ((N/NTHREADS) * (*id)) + 1, st_elementov);
        for (int i = 1; i < (st_elementov); i+=2)
        {   
            int j = ((N/NTHREADS) * (*id)) + i;
            if((j + 1) <= (st_elementov * ((*id) + 1)) && (j + 1) < N){
                primerjaj_in_zamenjaj(&pseznam[j], &pseznam[j+1], &lokalno_urejeno[*id], id, &j);
            }
        }
    } else {  //Nit je liha
        // printf("SEM LIHA NIT %d, I: %d, ST_ELEMENTOV: %d\n", *id, ((N/NTHREADS) * (*id)), st_elementov);
        for (int i = 0; i < st_elementov; i+=2)
        {
            int j = ((N/NTHREADS) * (*id)) + i;
            if((j + 1) <= ((st_elementov * ((*id) + 1)) + 1) && (j + 1) < N){
                primerjaj_in_zamenjaj(&pseznam[j], &pseznam[j+1], &lokalno_urejeno[*id], id, &j);
            }
        }
    }
    return NULL;
}

int doloci_st_elementov(int* id) {
    if (*id == (NTHREADS - 1)) {
        return N / NTHREADS + N % NTHREADS;
    } else {
        return N / NTHREADS;
    }
}

void izpisi_seznam(void){
    printf("\n   ");
    int coutner = 0;
    for (int i = 0; i < N; i++)
    {
        printf("%02d ", i);
        coutner++;
        if(coutner == 5){
            printf("| ");
            coutner = 0;
        }
    }
    coutner = 0;
    printf("\n   ");
    for(int i = 0; i < N; i++){
       printf("%02d ", *(pseznam + i));
       coutner++;
       if(coutner == 5){
           printf("| ");
           coutner = 0;
       }
    }
    printf("\n");
}
