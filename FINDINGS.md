# Darma v0: havainnot

## Mikä Darma on: RQ-SM

**Darma is a Resonant Quantum-Spirited Model (RQ-SM): it does not predict words, it resonates with what it remembers.**

- **Resonant:** vastaus syntyy resonanssista. Se, mikä on samassa vaiheessa muistin kanssa, vahvistuu, ja muu kumoutuu.
- **Quantum-Spirited:** tila, superpositio, interferenssi ja mittaus ovat Darman rakenne. Ne ovat matematiikkaa kompleksivektoreilla tavallisella prosessorilla, eivät kvanttilaitteistoa.
- **Model:** Darma ei ole kielimalli eikä neuroverkko. Se ei koulutu eikä tuota uusia lauseita. Se oppii yhdellä läpikäynnillä tallentamalla, ja kaikki sen sanat ovat sille kerrottuja tai ihmisten repliikkejä.

Mittaus: `cargo run --release -- eval` (D = 1024, siemen 2026). Tulokset: `results/eval_*.json`.

## Lyhyesti

Ydin toimii yhdessä asiassa ja epäonnistuu kahdessa.

- **Toimii:** monimerkityksisen sanan merkitys ratkeaa interferenssillä. Darma osuu oikeaan merkitykseen 86,6 %:ssa, kun yleisin merkitys osuu 73,4 %:ssa.
- **Ei toimi:** roolit koko oikeasta lauseesta. Oikein menee 5,6 %.
- **Ei toimi:** munasolu ei hylkää mitään. Järjettömistä lauseista hylätään 0 %.

Molemmat epäonnistumiset johtuvat samoista kahdesta asiasta. Oikea data on harvaa, ja rooliniput ovat kohinaisia. Kynnysten opettaminen harjoitusdatan alakvantiilista ei toimi, kun harjoituslauseet eivät muistuta toisiaan.

## Mitä mitattiin

Data on ladattu, eikä mitään ole keksitty. Aineistoja on kaksi, ja kummankin merkinnät ovat ihmisten tekemiä:

- **Roolit:** Universal Dependencies, English EWT. Mukana ovat klausuulit, joissa verbillä on substantiivisubjekti ennen ja substantiiviobjekti jälkeen. Harjoitusklausuuleja on 682 ja testiklausuuleja 77, ja jako on aineiston oma. Testiklausuulit tulevat 72 eri lauseesta.
- **Merkitykset:** CoarseWSD-20, jossa on 20 sanaa. Harjoituslauseita on 23 370 ja testilauseita 10 196, ja jako on aineiston oma.

Kaksi kohtaa poikkeaa alkuperäisestä suunnitelmasta:

- **Järjettömät lauseet ovat oikeita testiklausuuleja käännetyssä järjestyksessä,** esimerkiksi "slogans chanted crowds". Valmista aineistoa ei löytynyt, joten järjestyksen kääntäminen on tämän kokeen ainoa itse tehty osa.
- **Merkitysten kontekstivektorit niputettiin harjoitusdatan merkintöjen mukaan.** Ilman merkintää ei tiedetä, mihin merkitykseen kunkin lauseen naapurisanat kuuluvat. Testimerkintöjä käytettiin vain mittaukseen.

| Mittari | Darma | Vertailukohta: sanapussi ja yleisin merkitys |
|---|---|---|
| Roolit koko lauseesta, 72 lausetta | 5,6 % | 1,4 % |
| Roolit pelkästä kolmen sanan klausuulista, 77 | 94,8 % | – |
| Merkityksen valinta, 10 196 lausetta | 86,6 % | 73,4 % |
| Järjettömien hylkäys, 30 | 0 % | 96,7 % |
| Väärät hylkäykset, lauseet ja klausuulit | 0 % | – |
| Järjestysherkkyys, "a v p" vastaan "p v a" | samankaltaisuus 0,34 | – |
| Nopeus | 41 ms lauseelta munasolun läpi | – |
| Koulutus, yksi läpikäynti | 2,3 s | – |

Järjestysherkkyydessä tila muuttuu selvästi, kun roolit vaihtuvat. Samankaltaisuus jää kuitenkin opitun kalvokynnyksen, 0,065, yläpuolelle, joten ohjeen ehto "alle kynnyksen" ei täyty. Syy on kynnys, ei esitystapa.

## Mikä toimi

- **Vektoriydin.** Sidonta, purku ja superpositio toimivat odotetusti. Purku palauttaa alkuperäisen samankaltaisuudella 1,000. Yhteen summaan mahtuu 100 roolisidosta, joista 97 luetaan oikein.
- **Merkityksen valinta interferenssillä.** Darma voittaa yleisimmän merkityksen 14 sanassa 20:stä. Suurimmat erot ovat seal, 75 % vastaan 36 %, mole, 75 % vastaan 37 %, ja crane, 79 % vastaan 52 %. Vertailukohta voittaa kuudessa sanassa, ja viidessä niistä yksi merkitys hallitsee harjoitusdataa: pitcher 100 %, bank 96 %, deck 89 %, pound 86 % ja yard 84 %. Kuudes on digit, jossa yleisin merkitys kattaa 69 % ja testilauseita on vain 42.
- **Sanajärjestyksen oppiminen.** Järjestysnippu oppi yhdellä läpikäynnillä, että tekijä tulee ennen tekoa ja teko ennen kohdetta. Kun Darma saa pelkän klausuulin, roolit osuvat 94,8 %:ssa. Testi on helppo, koska vaihtoehtoja on kuusi, mutta se osoittaa järjestyksen kulkevan vektoreissa oikein.
- **Nopeus.** Koulutus vei 2,3 sekuntia, eikä mitään optimoitu.

## Mikä ei toiminut ja missä kerroksessa

Erittely koskee 72:ta testilausetta, joiden roolit Darma luki koko lauseesta.

| Kerros | Mitä tapahtui |
|---|---|
| Uinti eli siittiöiden järjestys | Oikea luenta on 30 vahvimman joukossa vain 30 lauseessa. Muissa 42:ssa se on hävinnyt jo ennen munasolua. |
| 1. Kalvo | Kalvo päästää läpi lähes kaiken. Oikeista luennoista läpi pääsi 29 / 30, mutta niin pääsivät väärätkin. |
| 2. Kilpailu | Rooleissa ei ole kilpailua, koska UD-datan sanoilla on yksi merkitys. Merkityksissä kilpailu toimii hyvin, katso yllä. |
| 3. Hedelmöitys | Hedelmöitys hyväksyy aina ensimmäisen kalvon läpäisseen, eli vahvimman siittiön, ja se on yleensä väärä. |

**Uinti.** Siittiön vahvuus luetaan rooli- ja järjestysnipuista interferenssillä. Kaksi asiaa pilaa sen:

1. **Data on harvaa.** 66 lauseessa 72:sta jokainen merkitty klausuuli sisältää sanan, jota ei ole koskaan nähty harjoituksessa samassa roolissa. Muistilla ei ole niistä tietoa.
2. **Rooliniput ovat kohinaisia.** Nippu on 682 satunnaisvektorin summa. Ennennäkemätön sana saa siitä kohinaa noin √(682 / 2D) ≈ 0,6 esiintymän verran, ja kohina on satunnaisesti positiivista. Siksi esimerkiksi lauseessa "Angry crowds chanted anti-American slogans in the western city" vahvin luenta on tekijä "angry", teko "chanted" ja kohde "in". Darma ei tiedä, että "in" ja "the" eivät koskaan ole tekijöitä tai kohteita, koska muisti näkee vain klausuulien kolme sanaa eikä koskaan sanoja, jotka eivät täytä roolia.

**Kalvo.** Kynnys opittiin ohjeen mukaan harjoitusklausuulien alakvantiilista, 5 %. Kullekin harjoitusklausuulille mitattiin, kuinka paljon se muistuttaa lähintä muuta klausuulia. Oikeassa datassa useimmat klausuulit eivät jaa yhtään sanaa samassa roolissa minkään muun kanssa. Siksi alakvantiili on lähes nolla, 0,065, ja kalvo päästää läpi kaiken, joka jakaa edes satunnaista kohinaa muistin kanssa.

**Hedelmöitys.** Itsetesti purkaa roolit rakenteesta, joka on tehty tarkoista atomeista. Purku onnistuu siksi aina selvällä marginaalilla, vaikka luenta olisi täysin väärä. Muistista palautettu tieto ei vedä tarpeeksi vastaan, koska muisti ei muistuta luentaa. Testi mittaa siis vektorien siisteyttä eikä ymmärrystä.

**Järjettömät lauseet.** Käännetty klausuuli, kuten "slogans chanted crowds", hyväksyttiin aina. Vain yksi 30:stä luettiin takaperin alkuperäiseksi klausuuliksi. Muut 29 hyväksyttiin sellaisenaan, slogans tekijänä. Kalvo ja hedelmöitys eivät hylänneet niitä samoista syistä kuin yllä. Vertailukohta hylkää 96,7 %, koska se hylkää aina, kun sanaa ei ole nähty roolissaan. Datan harvuuden takia se hylkäisi samalla myös lähes kaikki oikeat testilauseet.

## Kolme ehdotusta seuraavaksi

1. **Kynnykset satunnaisuutta vastaan eikä harjoitusjakaumasta.** Kalvon ja hedelmöityksen kynnys opitaan vertaamalla, kuinka paljon satunnainen luenta saa samasta muistista. Luenta pääsee läpi vain, jos se muistuttaa muistia selvästi enemmän kuin sattuma, esimerkiksi tasolla, jonka 95 % satunnaisluennoista alittaa. Kynnys on yhä opittu datasta, mutta se ei romahda nollaan, kun data on harvaa. Kohinan odotusarvo kannattaa samalla vähentää roolinippujen lukemista.
2. **Sanojen roolikelpoisuus kaikista UD-lauseista.** Nyt muisti näkee vain 682 klausuulin kolme sanaa. UD EWT:ssä on kuitenkin noin 12 500 merkittyä lausetta, ja niissä jokaisella sanalla on lauseenjäsen. Yhdellä läpikäynnillä jokainen sana saa nipun siitä, missä rooleissa se esiintyy ja missä ei. Silloin "in" ja "the" oppivat, etteivät ne ole koskaan tekijöitä. Lisäksi ennennäkemätön sana saa roolitietoa naapuriensa kautta, samalla kontekstiniputuksella kuin merkitykset. Tämä korjaisi uinnin, jossa 42 lausetta 72:sta hävisi.
3. **Hedelmöitys kilpailevaksi.** Itsetesti ei saa purkaa vain luennan omaa rakennetta. Voittaja ja toiseksi paras luenta yhdistetään samaan tilaan, ja vain se hyväksytään, jonka roolit purkautuvat selvästi vahvempina. Lisäksi puhdistusmuistin pitää kattaa koko sanasto eikä vain lauseen sanoja. Silloin "en ymmärrä" syntyy, kun kaksi luentaa ovat yhtä vahvoja tai kun muisti vetää eri suuntaan kuin luenta. Juuri niin kävisi järjettömälle lauseelle.

## Jatko: kolme ehdotusta kokeiltu

Mittaukset tehtiin samalla datalla ja samalla `eval`-komennolla. Kukin muutos mitattiin erikseen.

| Mittari | v0 | + ehdotus 2: roolikelpoisuus kaikista UD-lauseista | + ehdotus 1: kalvon kynnys sattumaa vastaan | + ehdotus 3: kilpaileva hedelmöitys, peruttu |
|---|---|---|---|---|
| Roolit koko lauseesta, 72 | 5,6 % | 45,8 % | 45,8 % | 0 % |
| Oikea luenta 30 vahvimman joukossa | 30 / 72 | 62 / 72 | 62 / 72 | 62 / 72 |
| Roolit pelkästä klausuulista, 77 | 94,8 % | 94,8 % | 64,9 % | 1,3 % |
| Järjettömien hylkäys, 30 | 0 % | 0 % | 13,3 % | 100 % |
| Väärät hylkäykset, lauseet | 0 % | 0 % | 0 % | 91,7 % |
| Väärät hylkäykset, klausuulit | 0 % | 0 % | 26,0 % | 96,1 % |
| Nopeus lauseelta | 41 ms | 41 ms | 28 ms | 2 000 ms |

Vertailukohta sai samalla saman tiedon kuin Darma, eli roolilaskurit kaikista UD-harjoituslauseista. Sen tulokset ovat nyt 4,2 % rooleissa ja 63,3 % järjettömien hylkäyksessä.

- **Ehdotus 2 toimi parhaiten.** Jokainen UD-harjoituslauseen sana, yhteensä 14 449 lausetta ja noin 200 000 sanaa, painoi roolinsa omaan nippuunsa, myös roolittomuuden. Lisäksi sanan naapurusto painettiin roolin kontekstiprototyyppiin. Uinti löytää nyt oikean luennan 62 lauseessa 72:sta, ja roolit osuvat 45,8 %:ssa, kun aiemmin osuma oli 5,6 %. Täytesanat kuten "in" ja "the" eivät enää nouse rooleihin.
- **Ehdotus 1 antoi kalvolle hampaat, mutta liian vähän.** Kalvo tarkistaa nyt, sopiiko jokainen sana sille annettuun rooliin. Kynnys on se taso, jonka 95 % satunnaisluennoista alittaa, eli 0,023. Järjettömistä hylätään 13,3 %, mutta samalla 26 % oikeista klausuuleista hylätään väärin. Kokonaisissa lauseissa väärät hylkäykset pysyvät nollassa, koska jokin luenta pääsee aina läpi.
- **Ehdotus 3 ei toiminut, ja se peruttiin.** Viisi vahvinta luentaa yhdistettiin uintivoimilla painotettuna, ja roolit purettiin koko sanastoa vastaan. Kynnyksen piti oppia satunnaisista luentajoukoista. Satunnaisetkin joukot purkautuivat kuitenkin selvästi, koska painotus antaa aina yhdelle luennalle lähes kaiken painon. Kynnys nousi 0,49:ään, ja melkein kaikki hylättiin. Marginaali mittaa siis painojen epätasaisuutta eikä ymmärrystä. Lisäksi koko sanaston puhdistus vei kaksi sekuntia lauseelta.

Hedelmöitys on siis edelleen kerros, joka ei tee työtään: sen itsetesti onnistuu aina. Seuraavaksi kannattaa kokeilla testiä, joka mittaa luennan ja muistin välistä ristiriitaa eikä vektorien siisteyttä. Esimerkki: kun luenta yhdistetään muistin palauttamaan tietoon, siirtyykö jonkin roolin purku toiseen sanaan?

## Hedelmöityksen korjaus: ristiriitatesti

Hedelmöitys on nyt testi siitä, onko luenta ristiriidassa ytimen tiedon kanssa. Sen itsetesti ei enää ole pelkkä vektorien siisteyden mittaus.

- **Muisti vastaa yksin.** Jokaisesta roolista muistilta kysytään, kuka lauseen sanoista kuuluu siihen. Vastaus perustuu siihen, mitä luennan muut sanat ovat tehneet ja kokeneet harjoitusdatassa. Esimerkiksi tekijän kohdalla kysytään, keitä tämän verbin tekijät ovat olleet.
- **Kilpailijat ovat vain rooliin sopivia sanoja,** eli niitä, jotka läpäisevät kalvon. Muuten täytesanat kuten "to" ja "of" voittivat aina, koska monen merkityksen summa osoittaa yleiseen suuntaan.
- **Muisti yleistää merkitysten avulla.** Jokainen sana sai merkitysvektorin naapureistaan yhdellä läpikäynnillä. Lähteinä olivat UD, CoarseWSD ja 10 miljoonaa sanaa WikiText-103:a, joka on merkitsemätöntä tekstiä. Jos verbiä ei ole nähty, muisti kysyy kolmelta merkitykseltään lähimmältä. Merkitykset ovat osin järkeviä: "killed" on lähellä sanoja "injured", "captured" ja "arrested".
- **Kynnys opitaan oikeista harjoitusklausuuleista.** Se on raja, jonka 95 % niistä ylittää, kun klausuulin oma tieto on poistettu muistista.

**Tulos: rakenne toimii, mutta muisti on liian heikko, eivätkä mittarit muuttuneet.**

| Havainto | Arvo |
|---|---|
| Muisti tuntee testiklausuulin tekijä–teko-parin | 5 / 77 |
| Muisti tuntee testiklausuulin teko–kohde-parin | 6 / 77 |
| Muisti tukee oikeaa järjestystä käännetyn yli | 45 / 77, kun sattuma antaisi noin 38 |
| Opittu kynnys | −0,46 |

Kynnys on negatiivinen, koska muisti suosii oikeissakin harjoitusklausuuleissa usein jotain muuta sanaa kuin oikeaa. Siksi testi päästää lähes kaiken läpi, ja järjettömistä hylätään yhä 13,3 %.

Hedelmöitys on nyt oikea mittari. Se kertoo, ettei muistin tieto vielä riitä erottamaan järkevää järjettömästä. Seuraava askel on siksi muistin vahvistaminen eikä hedelmöityksen muuttaminen. Tarvitaan enemmän tekstiä merkityksiin, esimerkiksi koko WikiText-103:n 100 miljoonaa sanaa, ja enemmän merkittyjä tekijä–teko–kohde-pareja.

## Miksi munasolu ei toimi: kerroksittainen diagnoosi

Komento `cargo run --release -- diagnose` mittaa kerrokset erikseen.

**1. Kalvo katsoo sanoja yksi kerrallaan, joten se ei näe roolien vaihtoa.**

| Mittaus, testiklausuulien kolme sanaa ilman lausetta | Tulos |
|---|---|
| Oikea luenta läpäisee kalvon | 50 / 77 |
| Käännetty luenta läpäisee kalvon | 41 / 77 |
| Oikea luenta sopii rooleihinsa paremmin kuin käännetty | 38 / 77, eli sattuman tasolla |
| Oikean tekijän sopivuus tekijäksi, keskimäärin | 0,141 |
| Oikean kohteen sopivuus tekijäksi, keskimäärin | 0,089 |

Sama substantiivi on usein sekä tekijä että kohde, joten sanan roolikelpoisuus ei kerro, kumpi on kumpi. Lisäksi kalvon kynnys opittiin satunnaisista luennoista, ja niissä 87 % sanoista ei ole minkään roolin sanoja. Kynnys erottaa siis hölynpölyn tyyliin "of the in" mutta ei roolien vaihtoa. Roolien vaihdon havaitseminen vaatii tietoa sanapareista, eli siitä, kuka tekee mitä. Se on hedelmöityksen tehtävä, ei kalvon.

**2. Hedelmöityksen mekanismi toimii, mutta se ei yleisty.**

Mittaus tehtiin 300 harjoitusklausuulilla, kolmella sanalla ilman lausetta. Muisti tukee oikeaa luentaa:
- 251 / 300, eli 84 %, kun muisti saa pitää klausuulien omat parit;
- 120 / 300, eli 40 %, kun ne poistetaan muistista.

Kun muisti on nähnyt parin, ristiriitatesti toimii. Kun pari on uusi, merkitysvektoreiden kautta tuleva yleistys ei kanna. Testidatan sanapareista muisti tuntee vain 5–6 / 77. Pullonkaula on siis tieto siitä, kuka tekee mitä, eikä hedelmöityksen rakenne.

**3. Mittarissa on pieni virhe.** Osa käännetyistä klausuuleista ei ole järjettömiä, esimerkiksi "money mean gifts" ja "case makes hubris". Valtaosa on kuitenkin järjettömiä, kuten "massacre condemn people", "talks demanded clerics" ja "slogans chanted crowds". Virhe on pieni verrattuna kalvon ja hedelmöityksen ongelmiin.

**Johtopäätös:** munasolu tarvitsee laajan muistin siitä, kuka tekee mitä kenelle. Ihmisten merkitsemiä pareja on vain noin 14 000 lauseesta, mikä on liian vähän. Lähin keino on, että Darma lukee itse merkitsemätöntä tekstiä, kuten WikiTextiä, ja painaa muistiin ne klausuulit, joista se on varma. Darma oppisi siis lukemalla. Silloin uusien parien kattavuus voisi nousta harjoitusdatan tasolle, jolla hedelmöitys jo toimii, 84 %:iin.

## Lukemalla oppiminen: kokeiltu

Darma lukee merkitsemätöntä WikiTextiä lause kerrallaan. Kun sen vahvin luenta voittaa toiseksi parhaan opitulla marginaalilla ja läpäisee kalvon, se painaa luennan parit muistiin. Marginaali opitaan UD-harjoituslauseista, ja tarkkuustavoite on 90 %. Lisäksi Darma painaa vain luennat, joiden kaikki kolme sanaa se tuntee. Komento `READ_PARAGRAPHS=N` valitsee, kuinka monta kappaletta luetaan. Oletus on 2 000.

Tulokset 2 000 kappaleella eli 8 921 lauseella, joiden lukeminen kesti 18 sekuntia:

| Versio | Painettuja klausuuleja | Uusien parien tuki, 300 harjoitusklausuulia |
|---|---|---|
| Ilman lukemista | – | 120 / 300 |
| Tarkkuustavoite 80 %, kaikki sanat | 1 306 | 108 / 300 |
| Tarkkuustavoite 90 %, vain tunnetut sanat | 147 | 119 / 300 |

- **80 %:n tavoitteella** painetuista oli silmämääräisesti noin puolet järkeviä, esimerkiksi "game / sold / units" ja "famitsu / enjoyed / story". Loput olivat roskaa, kuten "valkyria / chronicles / takeshi". Roskaparit heikensivät muistia.
- **90 %:n tavoitteella ja tunnetuilla sanoilla** noin kaksi kolmesta oli järkeviä, mutta niitä oli vähän. Muisti ei parantunut.
- **Kalibrointi on liian optimistinen.** Se tehdään harjoituslauseilla, jotka Darman roolimuisti jo tuntee. Lisäksi Wikipedia on eri tekstilaji kuin UD:n uutiset ja verkkotekstit.

Koko 10 miljoonan sanan WikiText antaisi samalla tahdilla noin 7 000 klausuulia, joista kolmannes olisi vääriä. Se ei riitä muuttamaan kattavuutta. Lukemalla oppiminen kantaa vasta, kun Darma lukee luotettavammin. Ongelma on kehämäinen: kunnollinen lukeminen vaatii sitä tietoa, jota lukemalla yritetään kerätä.

## Uusi ydin: kvanttihenkinen tila

Omistajan neljä riviä ovat nyt ytimen rakenne. Toteutus on tiedostoissa `src/quantum.rs` ja `src/qa.rs`.

| Rivi | Toteutus |
|---|---|
| tila = kompleksivektori | Kaikki kerrottu on superpositiona yhdessä tilassa. Lisäksi on avainrekisteri, jossa jokainen kerrottu lause on lomitettu jokaisen sanansa kanssa. |
| sana = operaattori | Sana kääntää tilan jokaisen luvun vaihetta, ja kääntöä edeltää yksi järjestysaskel. Operaatio on unitaarinen. Askeleen takia "dog bit man" ja "man bit dog" ovat eri tiloja. |
| kysymys = mittaus | Aukon jokainen ehdokas mitataan kysymyksen sanojen rekistereissä. Varmuus kertoo, missä määrin muisti pitää kysymyksen jonot samassa vaiheessa. |
| vastaus = interferenssi | Sopivat faktat vahvistavat toisiaan, ja muut kumoutuvat satunnaisten vaiheidensa takia. Vastaus annetaan vain, jos voittaja erottuu muista. Muuten Darma sanoo "en ymmärrä". |

**Mittaus:** Darmalle kerrottiin UD:n 682 oikeaa klausuulia, ja kysymykset tehtiin niistä korvaamalla tekijä sanalla "who" tai kohde sanalla "what". Dimensio oli 1 024.

| Mittari | Tulos |
|---|---|
| Kerrotut faktat, oikea vastaus | 95,1 % |
| Kerrotut faktat, väärä vastaus | 1,9 % |
| Kerrotut faktat, "en ymmärrä" | 3,0 % |
| Kertomattomat kysymykset, vastasi silti, 149 | 2,0 % |
| Väärin päin kysytyt, vastasi silti, 675 | 4,0 % |
| Nopeus | 14 ms kysymykseltä |
| Kertominen | 682 lausetta 0,05 sekunnissa |

**Mitä matkalla opittiin:**
1. **Yksi tila täyttyy nopeasti.** Kun 682 faktaa on yhdessä 1 024-ulotteisessa tilassa, signaali hukkuu kohinaan, ja oikein meni 0 %. Isompi tila auttaa hitaasti: dimensiolla 16 384 oikein meni 22,5 % 200 faktasta. Lomittunut avainrekisteri ratkaisi tämän. Kysymyksen sanat romahduttavat muistin niihin faktoihin, jotka kerrottiin samojen sanojen kanssa.
2. **Bornin todennäköisyys normalisoi ehdokkaiden kesken.** Silloin osittainen osuma näyttää yhtä varmalta kuin täysi. Päätös tehdään siksi jonokohtaisesta täsmäävyydestä, joka katkaistaan ykköseen, sekä siitä, kuinka paljon voittaja erottuu toiseksi parhaasta.
3. **Kalibrointi tehdään vain mittauksilla, jotka oikeasti tehtiin.** Jos kysymyksen jotain sanaa ei ole kerrottu, Darma ei mittaa mitään ja sanoo "en ymmärrä".
4. **Merkitysvektorit eivät auta tässä ytimessä.** Niiden kanssa oikein meni 93,9 %, ilman niitä 95,1 %. Ne tekevät sanoista toistensa kaltaisia, ja kohina lisääntyy.

Keskustelun voi käynnistää komennolla `cargo run --release -- chat`.

## Ytimen jatkokehitys

| Muutos | Klausuulikoe: kerrotut oikein / väärin päin vastasi | Nopeus |
|---|---|---|
| Lähtötilanne | 95,1 % / 4,0 % | 14 ms |
| Mittaus vain sanoista, jotka kerrottiin kysymyksen sanojen kanssa, ja z-raja opittuna väärin päin kysytyistä harjoituskysymyksistä | 95,0 % / 4,5 % | 33 µs |
| Mitataan vain rekistereissä, jotka erottavat täyden osuman kohinasta, kolmen keskihajonnan sääntö | 96,0 % / 3,6 % | 32 µs |
| Jonot 5 sanaan asti | 96,0 % / 5,1 % | 40 µs |

Väärin päin kysytyt mitataan parittomilla faktoilla. Raja opitaan parillisilla.

**Kyllä/ei-kysymykset.** Väite mitataan kokonaan, ja "kyllä" vaatii, että jokainen väitteen jono on muistissa.

| Väite | "Kyllä" |
|---|---|
| Kerrottu | 98,7 % |
| Väärin päin esitetty | 5,3 % |
| Kertomaton | 0 % |

**Päättely yhden askeleen yli.** Darma oppi UD:n merkinnöistä, mitkä sanat ovat yhdistäviä verbejä: am, are, be, been, being, is, 's, was ja were. Jos suora mittaus ei anna vastausta, Darma kysyy, mikä kysymyksen sana on, ja mittaa uudelleen vastaus sanan paikalla. Esimerkiksi "rex chased what" antaa vastauksen "the cat", koska rex is a dog.

**Muisti säilyy keskustelusta toiseen.** Kerrotut lauseet tallentuvat tiedostoon darma_memory.txt, ja ne kerrotaan uudelleen käynnistyksessä.

**Mittakaavakoe, `qa-sentences`.** Darmalle kerrottiin kaikki 14 449 UD-lausetta, noin 200 000 sanaa, 4–44 sekunnissa. Kysymykset tehtiin lauseiden omista sanoista, esimerkiksi "who killed a member".

| Asetus | Oikein | Väärin | Kertomattomiin vastasi |
|---|---|---|---|
| D 1 024, jonot 3 | 3,2 % | 0,4 % | 0 % |
| D 1 024, jonot 3, vain erottelukykyiset rekisterit | 31,8 % | 0 % | 0,8 % |
| D 4 096, jonot 3 | 38,0 % | 0,2 % | 2,5 % |
| D 16 384, jonot 3 | 29,4 % | 0 % | 0,8 % |
| D 4 096, jonot 4 | 53,2 % | 0,2 % | 0,8 % |
| D 4 096, jonot 5 | 58,2 % | 0,2 % | 1,7 % |

Ydin melkein ei koskaan vastaa väärin. Kun se ei ole varma, se sanoo "en ymmärrä". Kaksi asiaa rajoittaa sitä:
- **Rekisterin kapasiteetti.** Rekisteri erottaa osuman vain, jos sana esiintyy enintään noin 2D/9 jonossa. Dimensiolla 4 096 se tarkoittaa noin 20 lausetta sanaa kohti.
- **Kysymyksen ulottuvuus.** Pidemmät jonot tavoittavat kauempana olevat sanat.

Isompi dimensio ei yksin auta. Kun rekisterit tarkentuvat, myös aidot osittaiset osumat erottuvat, ja opittu raja nousee.

**Yksikkötestit:** 11 testiä menee läpi, vektoreille ja ytimelle.

**Kokeiltu ja hylätty:**
- **Parirekisterit.** Lause lomitettiin myös vierekkäisten sanaparien kanssa. Mittakaavakokeen tulos heikkeni 58 %:sta 12 %:iin, koska opittu raja nousi 0,85:een. Muutos peruttiin.
- **Merkitykset mittauksena, `qa-senses`.** CoarseWSD:n harjoituslauseet kerrottiin niin, että monimerkityksinen sana oli korvattu merkityksellään, ja testissä merkitys mitattiin aukkona. Oikein meni 49,1 %, kun vanha interferenssi saa 86,6 % ja yleisin merkitys 73,4 %. Uusi ydin muistaa tarkkoja sanajonoja, mutta merkityksen valinta vaatii yleistämistä lähiympäristöstä. Lisäksi 33 % jäi mittaamatta, koska ikkunassa oli uusi sana. Merkityksen valinta jää siksi lauseen sanojen superpositioksi, jota verrataan merkitysten konteksteihin. Sekin on interferenssimittaus, mutta eri kannassa.

## Kohti ymmärrystä: teko ja tunne

Keskustelukerros kopioi ihmisten vastauksia, ja omistajan tavoite on, että Darma ymmärtää eikä kopioi. Ensimmäinen askel on ymmärtää, mitä repliikki tekee ja mitä se tuntee.

DailyDialogin ihmiset ovat merkinneet jokaiseen vuoroon teon (kertominen, kysymys, pyyntö, lupaus) ja tunteen (ei tunnetta, viha, inho, pelko, ilo, suru, yllätys). Darma oppii ne yhdellä läpikäynnillä:
- **Vuoron tila** on sen 1–2 sanan jonojen tilojen summa. Jonot muodostetaan samoilla vaiheoperaattoreilla kuin ytimessä, ja harvinaiset jonot painavat enemmän.
- **Jokaisen teon ja tunteen prototyyppi** on sen vuorojen tilojen keskiarvo, josta on vähennetty kaikkien vuorojen yhteinen osa.
- **Uusi repliikki luetaan** sen prototyypin mukaan, jonka kanssa se resonoi vahvimmin. Luokan yleisyys painaa mukana. Painon suuruus valitaan validointiosalla, ja testiosalla mitataan vain kerran.

Merkintöjä käytetään vain ymmärtämisen opetteluun, eikä mitään vastausta kopioida.

| Testiosa, D 4 096 | Tarkkuus | Pelkkä yleisin luokka | F1-keskiarvo | Pelkkä yleisin luokka, F1 |
|---|---|---|---|---|
| Teko, 4 luokkaa, n 7 734 | 68,3 % | 45,6 % | 0,616 | 0,157 |
| Tunne, 7 luokkaa | 78,0 % | 81,7 % | 0,329 | 0,128 |

Dimensiolla 1 024 teon tarkkuus oli 63,9 % ja F1 0,575, tunteiden F1 0,320. Tunteiden tarkkuus jää yleisimmän luokan alle, koska 83 % vuoroista on tunteettomia. F1-keskiarvo on kuitenkin 2,6-kertainen. Yksittäisistä luokista ilo tunnistuu parhaiten, F1 0,45. Harvinaiset tunteet, kuten inho, jossa testivuoroja on vain 47, tunnistuvat heikosti.

Ensimmäisen kokeilun painot valitsin katsomalla testiosan tuloksia. Se oli vuoto, ja se korjattiin: luvut yllä on mitattu validointiosalla valituilla painoilla.

`cargo run --release -- --dim 4096 read "I'm so sad, my dog died"` näyttää, miten Darma lukee lauseen.

## Oma lause: vaiheet 2 ja 3, versio 1

`src/compose.rs`, komento `compose-eval`. Darma lukee repliikistä teon, tunteen ja aiheen. Sen jälkeen se toimii näin:
- **Tunteeseen** se vastaa niin kuin ihmiset siihen tunteeseen vastasivat.
- **Aiheeseen** se vastaa ihmisten muotilla, jossa on paikka aiheelle, esimerkiksi "What kind of ___ do you like?". Muotti valitaan saman teon ja tunteen repliikeistä sen mukaan, mikä alkuperäinen tilanne on lähimpänä. Aukkoon tulee käyttäjän aihe.
- **Muuten** se vastaa niin kuin tähän tekoon yleisimmin vastataan.

Opittu päätös, eli millä teolla ihmiset vastaavat, on järkevä: kysymykseen vastataan kertomalla ja pyyntöön lupaamalla.

| DailyDialog test | Hakuversio | Muotit, versio 1 | Aihe asiasanaksi ja muotit yleisiksi |
|---|---|---|---|
| F1 oikeaan vastaukseen, satunnainen 0,078 | 0,191 | 0,089 | 0,054 |
| Vastaus on sanasta sanaan harjoitusrepliikki | 100 % | 23,5 % | 49,5 % |

Esimerkkejä:
- "This is our rock-bottom price…" → "What kind of rock do you want"
- "It takes a long time to learn Chinese writing." → "Thank you"

**Miksi tulos on heikko:**
1. **Aihe on väärä.** Aiheeksi pitäisi valita substantiivilausekkeen pääsana: ilmauksessa "rock-bottom price" se on "price", ei "rock". Darmalla ei ole vielä lausekerakennetta. UD:ssa pääsanat on merkitty, joten Darma voi oppia ne sieltä.
2. **Darmalla ei ole omaa sisältöä.** Ihmisten vastauksissa on vastaajan omaa elämää: kokemuksia, mielipiteitä ja päätöksiä. Darma voi vain reagoida tunteeseen, kysyä aiheesta tai kertoa muistamansa. Siksi sen vastaukset jäävät väkisinkin kauas ihmisten vastauksista.
3. **Mittari ei sovi tähän.** Päällekkäisyys oikean vastauksen kanssa palkitsee kopiointia. Se ei mittaa, onko vastaus sopiva. Oikea mittari vaatisi ihmisen arvion, esimerkiksi omistajan.

Kun lyhyiden yleismuottien suosimista vähennettiin, kopiointi nousi. Syy on se, että vastaukset kuten "Thank you", "Ok" ja "Yes" ovat itse yleisiä repliikkejä.

## Oma lause, versio 2: pääsanat ja merkitys

- **Aihe.** Aihe valitaan nyt UD:n pääsanamerkinnöistä (heads.json). Sana kelpaa aiheeksi, jos se on yleensä substantiivilausekkeen pääsana eikä määrite, eikä sitä seuraa heti toinen asiasana. Esimerkiksi "chinese" on aina määrite, ja "price" on yleensä pääsana.
- **Muotti.** Muotti valitaan sen mukaan, kenen alkuperäisen repliikin tila resonoi merkitykseltään vahvimmin käyttäjän repliikin kanssa. Merkitysaallot tulevat keskustelumuistista.
- **Testi** kestää nyt noin 4 sekuntia, 150 keskustelua.

| DailyDialog test, 150 keskustelua | F1 oikeaan vastaukseen | Sanasta sanaan harjoitusrepliikki |
|---|---|---|
| Versio 2 | 0,073 | 43,8 % |
| Versio 1, koko testiosa | 0,054–0,089 | 23,5–49,5 % |
| Hakuversio | 0,210 | 100 % |

**Menetelmän raja:** muotti, jota ihmiset ovat käyttäneet monen eri aiheen kanssa, on lähes aina lyhyt, kuten "Which ___" tai "___ please". Pidemmät muotit esiintyvät vain kerran, ja ne ovat yksittäisiä lauseita eivätkä yleisiä muotteja. Yhden aukon muotit eivät siis yksin riitä oman lauseen tekemiseen. Seuraava askel olisi monen aukon rakenne, jossa on tekijä, teko ja kohde, ja jonka Darma oppisi UD:sta. Sisältö tulisi Darman omasta tiedosta, mutta tätä ei ole vielä rakennettu.

## Oma lause, versio 3: teko ja kohde

- **Lauseen ymmärrys.** Darma etsii repliikistä ensimmäisen verbin, eli UD:ssa verbiksi merkityn sanan, joka ei ole yhdistävä verbi, ja sen jälkeisen pääsanan. Esimerkiksi "I like football" antaa teon "like" ja kohteen "football".
- **Kaksipaikkaiset muotit.** Vastauksista, jotka toistavat sekä edellisen puhujan teon että kohteen, opitaan muotteja, esimerkiksi "What kind of [KOHDE] do you [TEKO]?". Darma täyttää ne vain samanmuotoisella verbillä. Muotojen päätteet (-ing, -ed, -s) ovat käsin kirjoitettu sääntö.
- **Data on harvaa.** 76 000 vuoroparista vain 201 toistaa sekä teon että kohteen. Yksikään muotti ei esiinny kahdesti, joten muotteja ei voi vaatia yleisiksi.

| DailyDialog test, 150 keskustelua | F1 | Kopioita |
|---|---|---|
| Versio 3, kaksipaikkaiset muotit | 0,097–0,100 | 31–33 % |
| Versio 2 | 0,073 | 44 % |
| Hakuversio | 0,210 | 100 % |

F1 on tähän asti paras oma lause, mutta lauseet ovat sekavia. Kerran nähty muotti on yksittäinen lause, josta on vaihdettu kaksi sanaa, ja muu alkuperäinen sisältö jää mukaan. Esimerkiksi "I bought a new car" saa vastauksen "That's funny my bought always loses ten car a day".

**Johtopäätös:** omaa lausetta ei voi koota ihmisten yksittäisistä lauseista. Se pitää koota rakenteista, joita on nähty monessa lauseessa: lauseen osien järjestyksestä ja siitä, mitkä sanat sopivat mihinkin paikkaan. Sisällön pitää tulla siitä, mitä Darma tietää. Tämä on kieliopin oppimista, ja se on seuraavan vaiheen ydin.
