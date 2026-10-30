# Darma v0: havainnot

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
- **Merkityksen valinta interferenssillä.** Darma voittaa yleisimmän merkityksen 14 sanassa 20:stä. Suurimmat erot ovat seal, 75 % vastaan 36 %, mole, 75 % vastaan 37 %, ja crane, 79 % vastaan 52 %. Vertailukohta voittaa vain sanoissa, joissa yksi merkitys on lähes ainoa: pitcher, jossa pelaaja on 100 % harjoitusdatasta, sekä bank, deck, pound ja digit.
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
