from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
adapter_repo = "asulova/hamlet-model"

# Load base model on CPU
model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load LoRA adapter
model = PeftModel.from_pretrained(model, adapter_repo)
model.eval()

# Run inference
prompt = (
    "INSTRUCTIONS: You are an assistant. "
   "Answer the question based strictly on the passages provided. "
    "You may slightly  expand, clarify, or rephrase to improve understanding,"
    "but do not invent facts not in the CONTEXT. "
    "Maintain the style of the model as learned during training."
    "If the answer is not present, reply: I can't find this in the context. "
    "CONTEXT:\n"
    "[Source 1: data/raw/hamlet.txt] none else near. Oph. I shall th' effect of this good lesson keep As watchman to my heart. But, good my brother, Do not as some ungracious pastors do, Show me the steep and thorny way to heaven, Whiles, like a puff'd and reckless libertine, Himself the primrose path of dalliance treads And recks not his own rede. Laer. O, fear me not! Enter Polonius. I stay too long. But here my father comes. A double blessing is a double grace; Occasion smiles upon a second leave. Pol. Yet here, Laertes? Aboard, aboard, for shame! The wind sits in the shoulder of your sail, And you are stay'd for. There- my blessing with thee! And these few precepts in thy memory Look thou character. Give thy thoughts no tongue, Nor any unproportion'd thought his act. Be thou familiar, but by no means vulgar: Those friends thou hast, and their adoption tried, Grapple them unto thy soul with hoops of steel; But do not dull thy palm with entertainment Of each new-hatch'd, unfledg'd comrade. Beware Of entrance to a quarrel; but\n\n"
    "[Source 2: data/raw/hamlet.txt] think meet To put an antic disposition on), That you, at such times seeing me, never shall, With arms encumb'red thus, or this head-shake, Or by pronouncing of some doubtful phrase, As 'Well, well, we know,' or 'We could, an if we would,' Or 'If we list to speak,' or 'There be, an if they might,' Or such ambiguous giving out, to note That you know aught of me- this is not to do, So grace and mercy at your most need help you, Swear. Ghost. [beneath] Swear. [They swear.] Ham. Rest, rest, perturbed spirit! So, gentlemen, With all my love I do commend me to you; And what so poor a man as Hamlet is May do t' express his love and friending to you, God willing, shall not lack. Let us go in together; And still your fingers on your lips, I pray. The time is out of joint. O cursed spite That ever I was born to set it right! Nay, come, let's go together. Exeunt. Act II. Scene I. Elsinore. A room in the house of Polonius. Enter Polonius and Reynaldo. Pol. Give him this money and these notes, Reyna …\n\n"
    "[Source 3: data/raw/hamlet.txt] nor stands it safe with us To let his madness range. Therefore prepare you; I your commission will forthwith dispatch, And he to England shall along with you. The terms of our estate may not endure Hazard so near us as doth hourly grow Out of his lunacies. Guil. We will ourselves provide. Most holy and religious fear it is To keep those many many bodies safe That live and feed upon your Majesty. Ros. The single and peculiar life is bound With all the strength and armour of the mind To keep itself from noyance; but much more That spirit upon whose weal depends and rests The lives of many. The cesse of majesty Dies not alone, but like a gulf doth draw What's near it with it. It is a massy wheel, Fix'd on the summit of the highest mount, To whose huge spokes ten thousand lesser things Are mortis'd and adjoin'd; which when it falls, Each small annexment, petty consequence, Attends the boist'rous ruin. Never alone Did the king sigh, but with a general groan. King. Arm you, I pray you, to th …\n\n"
    "QUESTION: What guidance does Polonius give Laertes about choosing and keeping friends?\n\n"
    "ANSWER:"
)
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
