#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include <stddef.h>

#define NTHREADS 32
#define N 40        // N naj bo vedno sodo število

#define __PRINT__

typedef struct {
    int id;
} args_t;

int* pseznam;
pthread_t nit[NTHREADS];
args_t arg[NTHREADS];
pthread_barrier_t prepreka;
bool konec = false;
bool lokalno_urejeno[NTHREADS];


void primerjaj_in_zamenjaj(int* pa, int* pb, bool *lokalno_urejeno);
void sodi_prehod(void);
void lihi_prehod(void);
void izpisi_seznam(void);
void* uredi_seznam(void* args);


int main(void){

    pseznam = (int*)malloc(N*sizeof(int));

    srand(time(NULL));
    for(int i = 0; i < N; i++){
        *(pseznam + i) = rand() % 100;
    }
#ifdef __PRINT__
    izpisi_seznam();
    printf("\n\n");
#endif

    // inicializiraj prepreko za NTHREADS niti:
    pthread_barrier_init(&prepreka, NULL, NTHREADS);

    // ustvarimo NTHREADS niti:
    for (size_t i = 0; i < NTHREADS; i++)
    {
        arg[i].id = i;
        pthread_create(
            &nit[i],
            NULL,
            uredi_seznam,
            (void*) &arg[i]);
    }


    // pocakajmo, da se vse niti zakljucijo:
    for (size_t i = 0; i < NTHREADS; i++)
    {
        pthread_join(nit[i], NULL);
    }
    
    return 0;
}


void primerjaj_in_zamenjaj(int* pa, int* pb, bool* lokalno_urejeno){
    int tmp;
    if(*pa > *pb){
        tmp = *pa;
        *pa = *pb;
        *pb = tmp;
        *lokalno_urejeno = false;
    }
}

void izpisi_seznam(void){
    for(int i = 0; i < N; i++){
       printf("%d ", *(pseznam + i));
    }
    printf("\n");
}


void* uredi_seznam(void* args) {

    args_t* pargumenti = (args_t*) args; 
    int id = pargumenti->id;

    while(!konec)
    {
        lokalno_urejeno[id] = true;
        // sodi prehod:
        pthread_barrier_wait(&prepreka);

        for (size_t j = id * 2; j < N; j = j + NTHREADS * 2)
        {
            // primerjaj elementa na indeksij j in j+1
            primerjaj_in_zamenjaj(pseznam+j, pseznam+j+1, &lokalno_urejeno[id]);
            //primerjaj_in_zamenjaj(&pseznam[j], &pseznam[j+1]);
        }

        pthread_barrier_wait(&prepreka);
#ifdef __PRINT__
        // samo nit 0 izpiše delno urejen seznam:
        if(id == 0){
            printf("   SODI: ");
            izpisi_seznam();
        }
        pthread_barrier_wait(&prepreka);
#endif
        
        // lihi prehod:
        for (size_t j = (id*2) + 1; j < N-1; j = j + NTHREADS * 2)
        {
            primerjaj_in_zamenjaj(pseznam+j, pseznam+j+1, &lokalno_urejeno[id]);
        }

        pthread_barrier_wait(&prepreka);

        // samo nit 0 izpiše delno urejen seznam:
        if(id == 0){
#ifdef __PRINT__
            printf("   LIHI: ");
            izpisi_seznam();
            printf("   ------------------------------\n");
#endif
        }
        pthread_barrier_wait(&prepreka);
        
        if (id == 0) {
            bool all_sorted = true;
            for (int k = 0; k < NTHREADS; k++) {
                if (!lokalno_urejeno[k]) {
                    all_sorted = false;
                    break;
                }
            }
            if (all_sorted) {
                konec = true;
                printf("Konec\n");
            }
        }

        pthread_barrier_wait(&prepreka);

        if (konec) {
            break;
        }
    }

    return NULL;
}