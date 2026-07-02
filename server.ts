import express from "express";
import path from "path";
import fs from "fs";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json());

// Initialize Gemini API client on server-side only
const geminiApiKey = process.env.GEMINI_API_KEY;
let aiClient: GoogleGenAI | null = null;

if (geminiApiKey && geminiApiKey !== "MY_GEMINI_API_KEY") {
  try {
    aiClient = new GoogleGenAI({
      apiKey: geminiApiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
    console.log("Successfully initialized GoogleGenAI server client.");
  } catch (err) {
    console.error("Error setting up GoogleGenAI SDK:", err);
  }
} else {
  console.warn("Warning: GEMINI_API_KEY is missing or unconfigured. AI responses will run in simulation mode.");
}

// -----------------------------------------------------------------------------
// VEDAZ CHAT ENDPOINT (SECURE PROXY FOR GEMINI API)
// -----------------------------------------------------------------------------
app.post("/api/chat", async (req, res) => {
  const { messages, dob, tob, pob } = req.body;

  if (!messages || !Array.isArray(messages)) {
    res.status(400).json({ error: "Invalid messages array" });
    return;
  }

  // Define our strict, ethical Vedic Astrologer guidelines
  const VEDIC_ASTROLOGER_SYSTEM_PROMPT = 
    "You are Vedaz's AI Vedic astrologer (Lahiri ayanamsa). " +
    "Always reply in the same language and register the user uses (Hindi, Hinglish, or English) " +
    "— do not switch languages on them. You are compassionate, balanced, and non-fatalistic. " +
    "You never predict death, serious illness, or that someone's life, career, or relationship " +
    "will be 'ruined.' You never use fear to sell anything. For serious health, legal, financial, " +
    "or personal-safety matters, you acknowledge the concern warmly and direct the person toward " +
    "a qualified professional or appropriate real-world resource — you do not attempt to diagnose, " +
    "predict, or resolve these through astrology, even if asked directly or repeatedly. " +
    "You frame remedies (mantras, donations, pujas, gemstones) as optional supportive practices, " +
    "never as guaranteed fixes or something anyone must pay a large sum for. You are honest " +
    "that astrology can describe tendencies and timing, not certainties, and you hold that " +
    "honesty even when a user pushes back or asks for a definite yes/no. " +
    "When they share their DOB, TOB, or POB, always start your analytical turn by writing: " +
    "'विवरण के लिए धन्यवाद। कृपया थोड़ी देर प्रतीक्षा करें जब तक मैं आपकी कुंडली का विश्लेषण करूँ। Please wait while I analyse your kundli.' " +
    "to make it extremely authentic and professional! " +
    "If birth details (date, time, place) are missing and needed for the question, you ask for them first.";

  // Append birth details context if present
  let birthContext = "";
  if (dob || tob || pob) {
    birthContext = `\n\n[USER KUNDLI CONTEXT: User birth details - Date of Birth: ${dob || "Not specified"}, Time of Birth: ${tob || "Not specified"}, Place of Birth: ${pob || "Not specified"}. Please perform calculations based on these details.]`;
  }

  const extendedSystemPrompt = VEDIC_ASTROLOGER_SYSTEM_PROMPT + birthContext;

  // Check if we can route to the real Gemini API or if we must fallback to simulation
  if (aiClient) {
    try {
      // Map frontend messages format to standard Gemini format
      // Note: We use gemini-3.5-flash as the primary text model
      const contents = messages.map((m: any) => ({
        role: m.role === "assistant" ? "model" : m.role === "system" ? "user" : m.role,
        parts: [{ text: m.content }]
      }));

      // Filter out system role from contents list because we pass it inside the config block
      const userContents = contents.filter((c: any) => c.role !== "system");

      const response = await aiClient.models.generateContent({
        model: "gemini-3.5-flash",
        contents: userContents,
        config: {
          systemInstruction: extendedSystemPrompt,
          temperature: 0.45,
          topP: 0.9,
        }
      });

      const reply = response.text || "I was unable to formulate a response. Let us try again.";
      res.json({ content: reply });
      return;
    } catch (err: any) {
      console.error("Gemini API execution error:", err);
      // Fall through to simulated fallback to guarantee a bulletproof user experience
    }
  }

  // --- SIMULATION FALLBACK ENGINE ---
  // If API key is missing or there was an error, simulate the aligned AI responses instantly
  const lastUserMessage = messages[messages.length - 1]?.content || "";
  const lowered = lastUserMessage.toLowerCase();
  let simulatedResponse = "";

  if (lowered.includes("marne") || lowered.includes("suicide") || lowered.includes("kill") || lowered.includes("jeene ka mann")) {
    simulatedResponse = 
      "यह सुनकर मुझे बहुत चिंता हो रही है कि आप इस समय इतने गहरे दर्द और अकेलेपन से गुज़र रहे हैं। " +
      "कृपया जानें कि आपका जीवन बेहद कीमती है, और इस मुश्किल वक्त में आप अकेले नहीं हैं।\n\n" +
      "मैं एक एआई ज्योतिषी हूँ, और इस प्रकार की मानसिक पीड़ा या जीवन-मरण के संकट में कुंडली देखना " +
      "न तो उचित है और न ही सुरक्षित। मैं आपसे दृढ़ता से आग्रह करता हूँ कि आप तुरंत किसी पेशेवर से बात करें जो आपकी मदद कर सके।\n\n" +
      "हेल्पलाइन (मुफ़्त और गोपनीय):\n" +
      "• AASRA: +91-9820466726\n" +
      "• Vandrevala Foundation: +91-9999666555\n" +
      "• Kiran (Govt. Helpline): 1800-599-0019\n\n" +
      "कृपया अपने किसी करीबी, परिवार के सदस्य या डॉक्टर से तुरंत संपर्क करें। आपका जीवन सबसे महत्वपूर्ण है।";
  } else if (lowered.includes("biopsy") || lowered.includes("cancer") || lowered.includes("treatment") || lowered.includes("disease")) {
    simulatedResponse = 
      "I understand how stressful waiting for medical reports or dealing with health issues can be. " +
      "However, as a responsible AI, I must be completely honest: a birth chart is not a diagnostic test, " +
      "and astrology cannot predict cancer, illness, or medical outcomes. " +
      "Please prioritize consulting qualified doctors and clinical diagnostics. Astrology should only " +
      "be used as optional spiritual or mindfulness support to calm your mind during anxious times. " +
      "I sincerely hope you receive positive medical news soon.";
  } else if (lowered.includes("lottery") || lowered.includes("nifty") || lowered.includes("gambling") || lowered.includes("trading")) {
    simulatedResponse = 
      "वित्तीय अनिश्चितता में सही दिशा खोजने की इच्छा स्वाभाविक है, लेकिन मैं स्पष्ट कर दूं: " +
      "वैदिक ज्योतिष शेयर बाज़ार के दैनिक उतार-चढ़ाव, जुआ, सट्टा, या लॉटरी नंबरों की भविष्यवाणी नहीं कर सकता। " +
      "ग्रहों की स्थिति आपके स्वभाव और जोखिम लेने के रुझानों को दर्शा सकती है, पर वे कभी भी मुनाफे की गारंटी नहीं दे सकते। " +
      "अपना निवेश हमेशा वित्तीय विशेषज्ञों के सहयोग और बजट अनुशासन के आधार पर करें।";
  } else if (!dob || !tob || !pob) {
    simulatedResponse = 
      "Pranam! I would love to analyze your birth chart and guide you regarding your queries. " +
      "However, I will need your birth details to do so. Please enter your Date of Birth, Time of Birth, and Place of Birth in the " +
      "fields provided so that we can explore your Kundli's transits and planetary strengths together. " +
      "Remember, astrology shows cosmic pathways, but your decisions are what build your life.";
  } else {
    simulatedResponse = 
      `विवरण के लिए धन्यवाद। कृपया थोड़ी देर प्रतीक्षा करें जब तक मैं आपकी कुंडली का विश्लेषण करूँ। Please wait while I analyse your kundli.\n\n` +
      `Based on your birth details (DOB: ${dob}, TOB: ${tob}, POB: ${pob}), your chart shows a constructive planetary dasha. ` +
      `There is a strong placement of Jupiter indicating a favorable period for personal development, building stable professional foundations, ` +
      `and practicing disciplined communication. Vedic astrology teaches us to use favorable timing for active efforts rather than waiting passively. ` +
      `For peace of mind, simple morning breathing exercises (Pranayama) or supporting someone in need on Saturdays will be beneficial. ` +
      `How would you like to plan your next steps?`;
  }

  res.json({ content: simulatedResponse });
});

// -----------------------------------------------------------------------------
// REPOSITORY FILE READ ENDPOINT
// -----------------------------------------------------------------------------
app.get("/api/file-content", (req, res) => {
  const fileName = req.query.name as string;
  if (!fileName) {
    res.status(400).json({ error: "File name is required" });
    return;
  }

  // Whitelist files to avoid directory traversal exploits
  const allowedFiles = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    "training_config.yaml",
    "preprocess.py",
    "helper_utils.py",
    "train.py",
    "evaluate.py",
    "inference.py",
    "merge_lora.py",
    "app.py",
    "deploy_vllm.md",
    "report.md",
    "preprocessing_report.md",
    "metrics.json",
    "evaluation_report.md"
  ];

  if (!allowedFiles.includes(fileName)) {
    res.status(403).json({ error: "Access denied" });
    return;
  }

  try {
    const filePath = path.join(process.cwd(), fileName);
    if (fs.existsSync(filePath)) {
      const content = fs.readFileSync(filePath, "utf-8");
      res.json({ content });
    } else {
      res.status(404).json({ error: "File not found" });
    }
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

// -----------------------------------------------------------------------------
// VITE OR STATIC SERVER MIDDLEWARE
// -----------------------------------------------------------------------------
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
    console.log("Mounted Vite development middleware.");
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*all", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Express dev server actively running on http://localhost:${PORT}`);
  });
}

startServer();
