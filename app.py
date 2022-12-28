import gradio as gr
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

t_1 = AutoTokenizer.from_pretrained("morenolq/bart-it-fanpage")
m_1 = AutoModelForSeq2SeqLM.from_pretrained("morenolq/bart-it-fanpage")

t_2 = AutoTokenizer.from_pretrained("morenolq/bart-it-ilpost")
m_2 = AutoModelForSeq2SeqLM.from_pretrained("morenolq/bart-it-ilpost")

t_3 = AutoTokenizer.from_pretrained("morenolq/bart-it-WITS")
m_3 = AutoModelForSeq2SeqLM.from_pretrained("morenolq/bart-it-WITS")

def predict(text, model="morenolq/bart-it-fanpage", max_length=64, do_sample=True, num_beams=1):

    if model == "morenolq/bart-it-fanpage":
        tokenizer = t_1
        model = m_1
    elif model == "morenolq/bart-it-ilpost":
        tokenizer = t_2
        model = m_2
    elif model == "morenolq/bart-it-WITS":
        tokenizer = t_3
        model = m_3

    text_summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

    text = text.strip()
    out_text = text_summarizer(text, max_length=max_length, do_sample=do_sample, num_beams=num_beams)[0]['summary_text']
    
    return out_text

iface = gr.Interface(
    fn=predict, 
    inputs=[
        gr.Textbox(lines=10, label="Input Text"),
        gr.Dropdown(["morenolq/bart-it-fanpage", "morenolq/bart-it-ilpost", "morenolq/bart-it-WITS"], label="Model", value="morenolq/bart-it-fanpage", show_label=True),
        gr.Slider(minimum=32, maximum=512, value=64, step=16, label="Max Length", show_label=True),
        gr.Checkbox(True, label="Beam Search", show_label=True),
        gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Num Beams", show_label=True)
    ],
    outputs=gr.Textbox(lines=2, label="Output Text"),
    description="Italian Text Summarization",
    examples=[
        [
            "Già la chiamano la tempesta mortale, e le autorità la definiscono «la più devastante tempesta di neve degli ultimi 50 anni». Anche se il peggio sembra essere passato, quello che la tormenta si lascia alle spalle è distruzione. Per ora, la conta dei morti si ferma a 50, di cui 27 nello Stato di New York. Tra tutte le immagini incredibili che arrivano da oltreoceano, colpiscono quelle delle centinaia di macchine abbandonate per le strade nell’area di Buffalo, nello Stato di New York, la zona più colpita. Le auto, in alcuni casi, si sono trasformate in vere e proprie trappole mortali . Come riporta il New York Times, Mark C. Poloncarz, a capo della contea di Erie, ha affermato che 14 dei 27 morti della sua zona, sono stati trovati per le strade, ma tre in macchina. Altri quattro sono morti perché non avevano il riscaldamento e tre sono morti per «eventi cardiaci» mentre rimuovevano la neve davanti alle loro case e ai negozi. Il sovrintendente della polizia dello Stato di New York, Steven Nigrelli, racconta che le autorità andavano di porta in porta, di macchina in macchina, a controllare la presenza di persone. Si continua a scavare per far riemergere veicoli sotterrati sotto la neve e il consiglio è ancora quello di non uscire.",
            "morenolq/bart-it-fanpage",
            64,
            False,
            1,
        ],
        [
            "Alla Camera la manovra ha concluso l'atterraggio nonostante le turbolenze, per usare la metafora del ministro dell'Economia, Giancarlo Giorgetti. Entro giovedì mattina Giorgia Meloni conta che il Senato approvi il via libera definitivo sul testo blindato. Poi nella conferenza stampa di fine anno tirerà un primo bilancio, consapevole che l'inizio del 2023 non si annuncia più semplice degli ultimi mesi. E dentro la maggioranza ripartirà in fretta il pressing dei partiti per migliorare alcune misure incluse nella legge di bilancio da 35 miliardi, e ripescare quelle rimaste fuori da una coperta troppo corta. Anche per questo, la prudenza ha consigliato all'esecutivo di mettere da parte un paio di miliardi di euro, alla fine non stanziati durante l'esame alla Camera fra ritocchi e retromarce. Saranno utili in vista di un nuovo decreto aiuti. Al di là delle misure contro il caro energia, nella manovra secondo la maggioranza ci sono una serie di novità che danno un segnale della visione dell'esecutivo, dalla flat tax per gli autonomi allo stop alla legge Fornero con l'introduzione di Quota 103, dalla stretta al Reddito di cittadinanza alla tregua fiscale. Qualche capitolo è saltato strada facendo, come la soglia di 60 euro per l'obbligo del Pos. O è stato depotenziato, come per le modifiche a Opzione donna: tanto che un ordine del giorno di FdI, approvato assieme alla manovra, impegna il governo ad ampliare la platea e anche la Lega ritiene che si potesse fare di più. È uno dei numerosi aspetti contestati dalle opposizioni, che hanno giudicato la legge di bilancio iniqua e piena di condoni e in questi giorni alla Camera potrebbero mettersi di traverso per ostacolare l'approvazione del dl rave entro il termine di venerdì, quando scadrà il primo decreto varato dal Consiglio dei ministri. In attesa di verificare gli effetti positivi del tetto al prezzo del gas definito dall'Ue, la crisi energetica resta, assieme alla congiuntura economica e al conflitto in Ucraina, fra i principali fattori di incertezza per lo scenario futuro. E fra le variabili da tenere sotto osservazione ai piani alti del governo ci sono anche i rapporti nella coalizione, soprattutto con Forza Italia. a premier e Silvio Berlusconi, raccontano nella maggioranza, si sono sentiti nei giorni prima di Natale per un 'rapido' scambio di auguri. Il clima fra i due da qualche tempo non è esattamente disteso. 'Tutto è bene quel che finisce bene', la sintesi degli azzurri, che nella manovra rivendicano l'aumento delle pensioni minime a 600 euro (mirando a raggiungere i mille euro nell'arco della legislatura) e la decontribuzione fino a 8mila euro per chi assume a tempo indeterminato dipendenti under 35. Resta il fatto che in FI ci si aspettava maggior coinvolgimento sin dall'inizio delle operazioni per costruire la prima legge di bilancio del governo. Senza contare che, nel clima caotico dell'esame a Montecitorio, si è anche sfiorato l'incidente interno alla maggioranza sullo scudo penale per i reati finanziari, fino all'ultimo dato per sicuro negli emendamenti dei relatori e poi saltato.",
            "morenolq/bart-it-fanpage",
            64,
            False,
            1,
        ],
    ]
)

iface.launch()