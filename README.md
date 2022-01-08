## Problema supusă
Tema 1 este compusă din 2 task-uri. Primul task conține extragerea datelor dintr-o imagine ce conține o tablă de sudoku. Rezultatul taskului să fie o matrice în care, pentru fiecare căsuță în care se află un număr, să fie pus un ‘x’, sau un ‘o’ în caz contrar. Pentru task-ul 2, este trebuie realizată extragerea datelor dintr-o tablă de jigsaw, iar, asemănător primului task, pe lângă datele căsuțelor ce conțin un număr sau nu,trebuie pus și id-ul zonei din care face parte.

## Soluția propusă

### Task 1

Pentru rezolvarea primului task, am folosit o serie de funcții integrate deja în librăria opencv.

Iterez prin fiecare imagine din path-ul specificat și încarc imaginea într-o matrice. Pentru a evita cazul în care marginea tablei de sudoku lovește marginea imaginii, adaug câte 50 de pixeli de culoare gri de-o parte si de alta a imaginii ca un fel de ‘padding’ pentru a preveni cazul în care nu poate fi rulat algoritmul de găsit contururi corespunzător. După acest padding, convertesc imaginea în grayscale pentru a lucra mai ușor, blurez pentru a scăpa de noise și aplic un threshold adaptiv pentru a scoate în evidentă marginile tablei și căsuțelor. Imaginea urmând să arate așa:

În continuare, pentru a identifica tabla de joc, apelez funcția cv.findContours pentru a identifica contururile din imagine și îl aleg pe cel mai mare (acesta fiind tabla). După ce am identificat tabla, ordonez colțurile astfel încât colțul din stânga sus să fie mereu primul, apoi dreapta-sus, dreapta-jos și stânga-jos.

După ordonarea punctelor, realizez o imagine nouă ce conține un ‘top down view’ al tablei pentru a avea linii drepte și pentru a evidenția cu cât mai multă ușurință pătratele. Realizez acest lucru printr-o transformare și warp-uire a imaginii inițiale.

După ce am realizat top down view-ul tablei de sudoku, iterez fiecare căsuță matematic (calculez total_width // 9 și total_height // 9 pentru a lua dimensiunea unei căsuțe) și aplic blur pentru a scăpa de noise și threshold pentru a evidenția și mai mult dacă acea căsuță conține informații. Pentru a verifica dacă o căsuță conține un număr, am calculat media valorilor pixelilor din căsuță. Cum, înainte de a aplica această idee am făcut un threshold, înseamnă că ori am pixeli complet albi, ori complet negri, astfel, pentru o căsuță ce nu conține un număr, media valorilor o să fie 255 (alb), iar pentru cele ce conțin informații, media valorilor o să fie mai mică (<255). Când fac această verificare completez în paralel un vector în care memorez id-urile căsuțelor ce conțin numere.

Într-un final, formez matricea răspuns, completând corespunzător fiecare valoare și salvez într-un fișier.


### Task 2

Pentru rezolvarea task-ului 2, mă folosesc în totalitate de ce este făcut la task-ul 1, dar adaug mici modificări și intrări noi în cod pentru rezolvarea problemei atribuirii unei zone.

După ce am extras tabla de joc în propria imagine, analog task-ului anterior, trebuie să extrag liniile ‘groase’ ce separă zonele jigsaw-ului. Pentru a realiza acest lucru, transform imaginea în grayscale, aplic un blur pentru a scăpa de noise, aplic un threshold adaptiv pentru a evidenția și mai bine toate liniile tablei. După toate acestea aplic funcția cv.morphologyEx cu parametrul ‘MORPH_OPEN’. Această funcție realizează 2 operații pe imagine - mai întâi erode (pentru a scăpa de liniile subțiri) și apoi dilate (pentru a readuce la aceeași mărime liniile groase). Pentru a scăpa de mici erori ce pot apărea pe marginile tablei, aplic un border de-alungul marginii.

După acest proces, aplic un findContours pentru a identifica fiecare dintre cele 9 zone și le dau fill cu alb pentru a realiza un canvas pe care ulterior îl voi colora.

Pentru a realiza delimitarea și identificarea zonelor, voi colora fiecare zonă cu o culoare unică prestabilita. Cele 9 culori sunt următoarele (numărul 0 este pentru debug, culorile au fost calculate folosind un tool ce returnează ‘maximum distinctive colors’):
Pentru a colora fiecare zonă, iterez fiecare căsuță (de la stânga la dreapta, de sus în jos) și verific dacă este colorată, dacă da - trec mai departe, dacă nu - colorez întreaga zonă din care face parte. La final tabla arătând astfel:
După realizarea acestei imagini, asemănător primului punct, iterez fiecare căsuță, verific dacă aceasta conține informații, folosind media valorilor pixelilor (această identificare se realizează pe imaginea cu tabla inițială dupa aplicarea de grayscale, blur și threshold, nu pe imaginea cu zonele) și, apoi, verific zona aferentă culorii din imaginea realizată mai sus. 

Salvez toate informațiile într-o matrice și scriu răspunsul în fișier.
