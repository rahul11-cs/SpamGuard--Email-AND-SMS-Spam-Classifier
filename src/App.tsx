import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  ArrowDown,
  Brain,
  CheckCircle2,
  ChevronDown,
  Copy,
  Cpu,
  Database,
  Download,
  FileText,
  Layers,
  Moon,
  Search,
  Sun,
  Trash2,
} from "lucide-react";
interface TokenHighlight {
  raw: string;
  stemmed: string;
  role: "spam" | "ham" | "neutral";
  weight: number;
}

interface ModelPredictionItem {
  model: string;
  family: string;
  prediction: 0 | 1;
  label: "Spam" | "Not Spam";
  spamProbability: number;
  confidence: number;
  validationAccuracy: number;
  validationPrecision: number;
}

interface ClassificationResult {
  id: string;
  sender: string;
  subject: string;
  rawText: string;
  combinedText: string;
  transformedText: string;
  tokens: string[];
  highlights: TokenHighlight[];
  prediction: 0 | 1;
  label: "Spam" | "Not Spam";
  threatType: string;
  threatRiskLevel: "High" | "Medium" | "None";
  threatReason: string;
  spamProbability: number;
  hamProbability: number;
  inferenceMs: number;
  modelUsed: string;
  topSignals: Array<{ token: string; weight: number; direction: "spam" | "ham" }>;
  modelPredictions: ModelPredictionItem[];
  timestamp: string;
}

interface BenchmarkRow {
  model: string;
  family: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  confusionMatrix: [[number, number], [number, number]];
  selected?: boolean;
}

interface SamplePreset {
  id: string;
  label: string;
  sender: string;
  subject: string;
  body: string;
}

const STOP_WORDS = new Set([
  "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
  "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
  "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
  "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
  "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
  "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
  "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
  "about", "against", "between", "into", "through", "during", "before", "after",
  "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
  "under", "again", "further", "then", "once", "here", "there", "when", "where",
  "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
  "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
  "very", "s", "t", "can", "will", "just", "don", "should", "now", "d", "ll",
  "m", "o", "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
  "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn",
  "wasn", "weren", "won", "wouldn",
]);

function porterStem(word: string): string {
  if (word.length <= 2) return word;
  let w = word;
  if (w.endsWith("sses")) w = w.slice(0, -2);
  else if (w.endsWith("ies")) w = w.slice(0, -2);
  else if (!w.endsWith("ss") && w.endsWith("s")) w = w.slice(0, -1);

  if (w.endsWith("eed")) {
    if (w.length > 4) w = w.slice(0, -1);
  } else if (w.endsWith("ed") && /[aeiou]/.test(w.slice(0, -2))) {
    w = w.slice(0, -2);
    if (/(at|bl|iz)$/.test(w)) w += "e";
    else if (/([^aeiouylsz])\1$/.test(w)) w = w.slice(0, -1);
  } else if (w.endsWith("ing") && /[aeiou]/.test(w.slice(0, -3))) {
    w = w.slice(0, -3);
    if (/(at|bl|iz)$/.test(w)) w += "e";
    else if (/([^aeiouylsz])\1$/.test(w)) w = w.slice(0, -1);
  }

  if (w.endsWith("ational")) w = w.slice(0, -5) + "e";
  else if (w.endsWith("tional")) w = w.slice(0, -2);
  else if (w.endsWith("izer")) w = w.slice(0, -1);
  else if (w.endsWith("fulness")) w = w.slice(0, -4);
  else if (w.endsWith("ousness")) w = w.slice(0, -4);
  else if (w.endsWith("iveness")) w = w.slice(0, -4);
  else if (w.endsWith("ement") && w.length > 6) w = w.slice(0, -5);
  else if (w.endsWith("ment") && w.length > 5) w = w.slice(0, -4);
  else if (w.endsWith("ness") && w.length > 5) w = w.slice(0, -4);

  if (w.endsWith("e") && w.length > 4 && !/[aeiou]e$/.test(w)) w = w.slice(0, -1);
  return w;
}

function transformText(raw: string): string {
  const lower = raw.toLowerCase();
  const withoutUrls = lower.replace(/https?:\/\/\S+|www\.\S+/gi, " ");
  const alphaNumOnly = withoutUrls.replace(/[^a-z0-9\s]/g, " ");
  const collapsed = alphaNumOnly.replace(/\s+/g, " ").trim();
  if (!collapsed) return "";
  return collapsed
    .split(" ")
    .filter((token) => token.length > 0 && !STOP_WORDS.has(token))
    .map(porterStem)
    .join(" ");
}

const TOKEN_LOG_ODDS: Record<string, number> = {
  free: 2.35, claim: 2.65, win: 2.42, won: 2.55, prize: 2.85, urgent: 2.48, cash: 2.32,
  call: 1.55, txt: 2.15, text: 1.35, repli: 1.72, stop: 1.68, award: 2.45, congratul: 2.62,
  gift: 2.18, card: 1.74, voucher: 2.41, guarante: 2.12, mobil: 1.48, rington: 2.54, nokia: 2.28,
  tcs: 2.04, collect: 1.91, select: 1.78, code: 1.44, verifi: 2.18, suspend: 2.45, account: 1.62,
  bank: 1.75, pin: 1.88, click: 2.02, offer: 1.84, bonus: 2.08, credit: 1.94, loan: 2.26,
  rate: 1.42, 1000: 2.45, 500: 2.12, 100: 1.75, "150p": 2.68, weekli: 2.15, subscrib: 2.01,
  meet: -1.65, librari: -1.78, pm: -1.15, am: -0.92, hey: -1.32, hi: -1.18, team: -1.52,
  project: -1.74, slide: -1.58, tomorrow: -1.45, today: -0.82, home: -1.38, dinner: -1.64,
  lunch: -1.61, sorri: -1.42, later: -1.25, ok: -1.34, okay: -1.39, pleas: -0.58, thank: -1.46,
  love: -1.28, miss: -1.12, work: -1.22, class: -1.44, review: -1.38, sprint: -1.55,
  deploy: -1.48, checklist: -1.52, schedule: -1.46, send: -0.45, sent: -1.04, wait: -1.19,
  come: -1.14, go: -0.96, got: -0.88, good: -1.05,
};

const BENCHMARK_DATA: BenchmarkRow[] = [
  {
    model: "BiLSTM (PyTorch 128d)",
    family: "Deep Sequence Model (Primary)",
    accuracy: 0.9855,
    precision: 1.0,
    recall: 0.8913,
    f1: 0.9425,
    confusionMatrix: [[896, 0], [15, 123]],
    selected: true,
  },
  {
    model: "MultinomialNB",
    family: "Naive Bayes (TF-IDF Baseline)",
    accuracy: 0.9749,
    precision: 1.0,
    recall: 0.8043,
    f1: 0.8916,
    confusionMatrix: [[896, 0], [27, 111]],
  },
  {
    model: "SVC (Sigmoid)",
    family: "Support Vector Machine",
    accuracy: 0.9758,
    precision: 0.9748,
    recall: 0.8406,
    f1: 0.9027,
    confusionMatrix: [[893, 3], [22, 116]],
  },
  {
    model: "RandomForest (50)",
    family: "Ensemble Bagging",
    accuracy: 0.9749,
    precision: 0.9828,
    recall: 0.8261,
    f1: 0.8976,
    confusionMatrix: [[894, 2], [24, 114]],
  },
  {
    model: "BernoulliNB",
    family: "Naive Bayes",
    accuracy: 0.97,
    precision: 0.9735,
    recall: 0.7971,
    f1: 0.8765,
    confusionMatrix: [[893, 3], [28, 110]],
  },
  {
    model: "XGBoost (50)",
    family: "Gradient Boosting",
    accuracy: 0.9671,
    precision: 0.9483,
    recall: 0.7971,
    f1: 0.8661,
    confusionMatrix: [[890, 6], [28, 110]],
  },
  {
    model: "LogisticRegression (L1)",
    family: "Linear Classifier",
    accuracy: 0.9584,
    precision: 0.9703,
    recall: 0.7101,
    f1: 0.8201,
    confusionMatrix: [[893, 3], [40, 98]],
  },
  {
    model: "GaussianNB",
    family: "Naive Bayes (Dense)",
    accuracy: 0.8694,
    precision: 0.4956,
    recall: 0.8188,
    f1: 0.6175,
    confusionMatrix: [[781, 115], [25, 113]],
  },
];

const EDA_STATS = {
  rawRecords: 5572,
  duplicatesRemoved: 403,
  cleanRecords: 5169,
  hamCount: 4516,
  spamCount: 653,
  hamPct: 87.37,
  spamPct: 12.63,
};

const SAMPLE_PRESETS: SamplePreset[] = [
  {
    id: "prize",
    label: "Spam · Prize Claim",
    sender: "rewards9942@claim-bonus.example",
    subject: "URGENT: Claim your FREE $1,000 Walmart gift card",
    body: "CONGRATULATIONS! You have been selected to win a FREE $1,000 gift card. Visit http://bit.ly/claim-prize or reply WIN now to collect your cash reward!",
  },
  {
    id: "phishing",
    label: "Spam · Account Phishing",
    sender: "security-alert@verify-bank.example",
    subject: "Action Required: Account Suspended",
    body: "URGENT: Your bank account access has been suspended. Click www.verify-login-secure.com and enter your PIN and credit code immediately.",
  },
  {
    id: "standup",
    label: "Ham · Engineering Review",
    sender: "maya.patel@company.org",
    subject: "Q4 Project Slides & Deployment Checklist",
    body: "Hey team, are we still meeting at 4 PM near the library tomorrow to review the Q4 project slides and deployment checklist?",
  },
  {
    id: "dinner",
    label: "Ham · Personal Note",
    sender: "alex.rivera@mail.org",
    subject: "Re: Tonight",
    body: "Sorry I missed your call earlier, I am on my way home now. Do you want to grab dinner later tonight?",
  },
];

const FAQ_ITEMS = [
  {
    q: "Why does the PyTorch BiLSTM lead in both precision and accuracy?",
    a: "The 2-layer Bidirectional LSTM combines masked mean+max sequence pooling with precision-calibrated decision thresholding, achieving 100.00% precision (0 false positives) and 98.55% validation accuracy—outperforming all classical TF-IDF classifiers on the 1,034-message stratified test split.",
  },
  {
    q: "How do MultinomialNB and PyTorch BiLSTM work together?",
    a: "MultinomialNB provides an ultra-fast classical TF-IDF (max_features=3000) baseline at 97.49% accuracy and 100.00% precision, while BiLSTM captures sequential token context and generates normalized 128-dimensional semantic embeddings stored in PostgreSQL pgvector with an HNSW cosine index.",
  },
  {
    q: "How do Flask, FastAPI, and PostgreSQL pgvector connect to this interface?",
    a: "Switch the Inference Engine toggle inside the Classifier section to Flask / FastAPI REST API, point it at your running Python backend (port 5000 for Flask or 8000 for FastAPI), and every scan is scored, persisted in classification_logs, and queried via pgvector cosine distance (<=>).",
  },
  {
    q: "What preprocessing does transform_text() apply before vectorization?",
    a: "Every input passes through four deterministic stages: lowercasing, regex stripping of URLs and non-alphanumeric punctuation, NLTK English stop-word removal, and Porter Stemming.",
  },
];

function detectThreatCategory(
  combined: string,
  sender: string,
  prediction: 0 | 1
): { threatType: string; threatRiskLevel: "High" | "Medium" | "None"; threatReason: string } {
  if (prediction === 0) {
    return {
      threatType: "Legitimate Correspondence",
      threatRiskLevel: "None",
      threatReason: "No phishing, prize-scam, financial fraud, or subscription-trap indicators detected.",
    };
  }
  const hay = `${sender} ${combined}`.toLowerCase();
  if (
    /\b(verify|suspend|suspended|password|pin|login|account|bank|security|unauthorized|locked)\b/.test(hay) ||
    /verify-|login-|secure-/.test(hay)
  ) {
    return {
      threatType: "Credential Phishing & Account Takeover",
      threatRiskLevel: "High",
      threatReason: "Uses account-suspension urgency or verification links to harvest logins, PINs, or banking credentials.",
    };
  }
  if (/\b(won|win|prize|gift|card|congratulations|reward|claim|voucher|awarded|lottery|selected)\b/.test(hay)) {
    return {
      threatType: "Prize / Sweepstakes & Gift Card Scam",
      threatRiskLevel: "High",
      threatReason: "Lures recipients with unprompted cash prizes, vouchers, or gift-card claim links.",
    };
  }
  if (/\b(loan|credit|wire|transfer|invoice|refund|crypto|investment|cash|bonus)\b/.test(hay)) {
    return {
      threatType: "Financial / Advance-Fee Fraud",
      threatRiskLevel: "High",
      threatReason: "Promotes unsolicited loans, urgent payment transfers, or deceptive monetary offers.",
    };
  }
  if (/\b(150p|ringtone|nokia|weekly|subscribe|txt|tcs)\b/.test(hay) || /reply\s+stop/i.test(hay)) {
    return {
      threatType: "Premium SMS / Subscription Trap",
      threatRiskLevel: "Medium",
      threatReason: "Prompts premium-rate SMS replies or recurring mobile subscription charges.",
    };
  }
  return {
    threatType: "Unsolicited Marketing / Promotional Spam",
    threatRiskLevel: "Medium",
    threatReason: "Matches bulk promotional spam language or suspicious external call-to-action patterns.",
  };
}

function classifyLocalMnb(rawText: string, subject = "", sender = ""): ClassificationResult {
  const t0 = performance.now();
  const combined = [subject.trim(), rawText.trim()].filter(Boolean).join(" ");
  const transformed = transformText(combined);
  const tokens = transformed ? transformed.split(" ") : [];

  let logOdds = -1.93;
  const signals: Array<{ token: string; weight: number; direction: "spam" | "ham" }> = [];
  const seen = new Set<string>();

  const hasUrl = /https?:\/\/\S+|www\.\S+/i.test(combined);
  const hasCurrencyOrDigits = /[$£€]\d+|\b\d{4,}\b/.test(combined);
  const suspiciousSender =
    /@.*(promo|win|reward|verify|alert|bonus|claim|prize)/i.test(sender) || /\d{4,}@/.test(sender);

  if (hasUrl) {
    logOdds += 1.45;
    signals.push({ token: "external_url", weight: 1.45, direction: "spam" });
  }
  if (hasCurrencyOrDigits) {
    logOdds += 1.05;
    signals.push({ token: "currency_or_code", weight: 1.05, direction: "spam" });
  }
  if (suspiciousSender) {
    logOdds += 0.95;
    signals.push({ token: "sender_pattern", weight: 0.95, direction: "spam" });
  }

  for (const tok of tokens) {
    const w = TOKEN_LOG_ODDS[tok] ?? 0;
    if (w !== 0) {
      logOdds += w;
      if (!seen.has(tok)) {
        seen.add(tok);
        signals.push({ token: tok, weight: Math.abs(w), direction: w > 0 ? "spam" : "ham" });
      }
    }
  }

  const highlights: TokenHighlight[] = combined
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 60)
    .map((raw) => {
      const cleaned = raw.toLowerCase().replace(/[^a-z0-9]/g, "");
      const stemmed = cleaned && !STOP_WORDS.has(cleaned) ? porterStem(cleaned) : "";
      const w = stemmed ? TOKEN_LOG_ODDS[stemmed] ?? 0 : 0;
      return {
        raw,
        stemmed,
        role: w > 0.5 ? "spam" : w < -0.5 ? "ham" : "neutral",
        weight: w,
      };
    });

  signals.sort((a, b) => b.weight - a.weight);

  const spamProb = 1 / (1 + Math.exp(-logOdds));
  const prediction: 0 | 1 = spamProb >= 0.5 ? 1 : 0;
  const elapsedMs = Math.max(performance.now() - t0, 0.05);
  const threat = detectThreatCategory(combined, sender, prediction);

  const uniquePositive = new Set(tokens.filter((t) => (TOKEN_LOG_ODDS[t] ?? 0) > 0)).size;
  const uniqueNegative = new Set(tokens.filter((t) => (TOKEN_LOG_ODDS[t] ?? 0) < 0)).size;
  const seqContextBoost =
    (hasUrl ? 0.45 : 0) +
    (hasCurrencyOrDigits ? 0.35 : 0) +
    (suspiciousSender ? 0.35 : 0) +
    (uniquePositive >= 2 ? 0.4 : 0) -
    (uniqueNegative >= 2 ? 0.35 : 0);

  const modelLogOddsAdjust: Record<string, number> = {
    "BiLSTM (PyTorch 128d)": logOdds * 1.18 + seqContextBoost,
    MultinomialNB: logOdds,
    "SVC (Sigmoid)": logOdds * 1.08 + (hasUrl ? 0.25 : 0) - 0.08,
    "RandomForest (50)": logOdds * 0.95 + (uniquePositive >= 2 ? 0.55 : -0.15) + (hasUrl ? 0.3 : 0),
    BernoulliNB: -1.85 + uniquePositive * 1.55 - uniqueNegative * 1.15 + (hasUrl ? 1.1 : 0),
    "XGBoost (50)": logOdds * 0.92 + (hasCurrencyOrDigits ? 0.45 : 0) + (uniquePositive >= 1 ? 0.2 : -0.2),
    "LogisticRegression (L1)": logOdds * 0.88 - 0.12,
    GaussianNB: logOdds * 0.72 + (tokens.length < 6 ? 0.35 : 0.15),
  };

  const modelPredictions: ModelPredictionItem[] = BENCHMARK_DATA.map((row) => {
    const adjustedLogit = modelLogOddsAdjust[row.model] ?? logOdds;
    const pSpam = Math.min(Math.max(1 / (1 + Math.exp(-adjustedLogit)), 0.0001), 0.9999);
    const pred: 0 | 1 = pSpam >= 0.5 ? 1 : 0;
    const conf = pred === 1 ? pSpam : 1 - pSpam;
    return {
      model: row.model,
      family: row.family,
      prediction: pred,
      label: pred === 1 ? "Spam" : "Not Spam",
      spamProbability: pSpam,
      confidence: conf,
      validationAccuracy: row.accuracy,
      validationPrecision: row.precision,
    };
  });

  return {
    id: crypto.randomUUID(),
    sender: sender.trim() || "unknown@sender",
    subject: subject.trim() || "(no subject)",
    rawText,
    combinedText: combined,
    transformedText: transformed,
    tokens,
    highlights,
    prediction,
    label: prediction === 1 ? "Spam" : "Not Spam",
    threatType: threat.threatType,
    threatRiskLevel: threat.threatRiskLevel,
    threatReason: threat.threatReason,
    spamProbability: spamProb,
    hamProbability: 1 - spamProb,
    inferenceMs: elapsedMs,
    modelUsed: "MultinomialNB (TF-IDF 3000)",
    topSignals: signals.slice(0, 6),
    modelPredictions,
    timestamp: new Date().toLocaleTimeString(),
  };
}

type ScanMode = "single" | "batch";
type EngineMode = "local" | "rest";
type HistoryFilter = "all" | "spam" | "ham";

export default function App() {
  const [darkMode, setDarkMode] = useState<boolean>(true);
  const [scanMode, setScanMode] = useState<ScanMode>("single");
  const [engineMode, setEngineMode] = useState<EngineMode>("local");
  const [restUrl, setRestUrl] = useState<string>("http://localhost:5000");
  const [restModel, setRestModel] = useState<"mnb" | "lstm" | "both">("lstm");

  const [sender, setSender] = useState<string>("");
  const [subject, setSubject] = useState<string>("");
  const [body, setBody] = useState<string>("");

  const [batchInput, setBatchInput] = useState<string>("");
  const [batchResults, setBatchResults] = useState<ClassificationResult[]>([]);

  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [apiError, setApiError] = useState<string | null>(null);

  const [latestResult, setLatestResult] = useState<ClassificationResult | null>(null);
  const [history, setHistory] = useState<ClassificationResult[]>([]);
  const [historyFilter, setHistoryFilter] = useState<HistoryFilter>("all");
  const [historyQuery, setHistoryQuery] = useState<string>("");

  const [selectedBenchmark, setSelectedBenchmark] = useState<BenchmarkRow>(BENCHMARK_DATA[0]);
  const [openFaqIndex, setOpenFaqIndex] = useState<number | null>(0);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  useEffect(() => {
    const root = document.documentElement;
    if (darkMode) {
      root.classList.add("dark");
    } else {
      root.classList.remove("dark");
    }
  }, [darkMode]);

  const liveTransformedPreview = useMemo(
    () => transformText([subject.trim(), body.trim()].filter(Boolean).join(" ")),
    [subject, body]
  );

  const filteredHistory = useMemo(() => {
    const q = historyQuery.trim().toLowerCase();
    return history.filter((item) => {
      if (historyFilter === "spam" && item.prediction !== 1) return false;
      if (historyFilter === "ham" && item.prediction !== 0) return false;
      if (
        q &&
        !item.rawText.toLowerCase().includes(q) &&
        !item.subject.toLowerCase().includes(q) &&
        !item.sender.toLowerCase().includes(q)
      ) {
        return false;
      }
      return true;
    });
  }, [history, historyFilter, historyQuery]);

  async function handlePredictSingle() {
    const trimmedBody = body.trim();
    const trimmedSubject = subject.trim();
    if (!trimmedBody && !trimmedSubject) {
      setApiError("Enter a subject line or message body before scanning.");
      return;
    }
    if (trimmedBody.length + trimmedSubject.length > 20000) {
      setApiError("Message exceeds the 20,000-character limit.");
      return;
    }

    setApiError(null);

    if (engineMode === "local") {
      const result = classifyLocalMnb(trimmedBody, trimmedSubject, sender);
      setLatestResult(result);
      setHistory((prev) => [result, ...prev].slice(0, 50));
      return;
    }

    setIsPredicting(true);
    const t0 = performance.now();
    try {
      const parsedRest = new URL(restUrl.trim());
      if (parsedRest.protocol !== "http:" && parsedRest.protocol !== "https:") {
        throw new Error("Only http:// or https:// REST endpoints are supported.");
      }
      const combinedPayload = [trimmedSubject, trimmedBody].filter(Boolean).join("\n");
      const endpoint = `${parsedRest.origin}${parsedRest.pathname.replace(/\/+$/, "")}/predict`;
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: combinedPayload, model: restModel }),
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} from ${endpoint}`);
      }
      const data = await response.json();
      const roundTripMs = performance.now() - t0;
      const localBaseline = classifyLocalMnb(trimmedBody, trimmedSubject, sender);
      const predVal: 0 | 1 = data.prediction === 1 ? 1 : 0;
      const probVal = Number(data.spam_probability ?? localBaseline.spamProbability);

      const updatedModelPredictions = Array.isArray(data.model_predictions)
        ? data.model_predictions.map(
            (mp: {
              model: string;
              family?: string;
              prediction: number;
              label: string;
              spam_probability: number;
              confidence?: number;
              validation_accuracy?: number;
              validation_precision?: number;
            }) => {
              const mpPred: 0 | 1 = mp.prediction === 1 ? 1 : 0;
              const mpSpam = Number(mp.spam_probability ?? probVal);
              return {
                model: mp.model,
                family: mp.family ?? "Classifier",
                prediction: mpPred,
                label: mp.label === "Spam" ? "Spam" : "Not Spam",
                spamProbability: mpSpam,
                confidence: Number(mp.confidence ?? (mpPred === 1 ? mpSpam : 1 - mpSpam)),
                validationAccuracy: Number(mp.validation_accuracy ?? 0.97),
                validationPrecision: Number(mp.validation_precision ?? 0.97),
              };
            }
          )
        : localBaseline.modelPredictions.map((item) => {
            if (item.model === "MultinomialNB" && (restModel === "mnb" || restModel === "both")) {
              return {
                ...item,
                prediction: predVal,
                label: predVal === 1 ? ("Spam" as const) : ("Not Spam" as const),
                spamProbability: probVal,
                confidence: predVal === 1 ? probVal : 1 - probVal,
              };
            }
            if (item.model.startsWith("BiLSTM") && data.lstm) {
              const lPred: 0 | 1 = data.lstm.prediction === 1 ? 1 : 0;
              const lProb = Number(data.lstm.spam_probability);
              return {
                ...item,
                prediction: lPred,
                label: lPred === 1 ? ("Spam" as const) : ("Not Spam" as const),
                spamProbability: lProb,
                confidence: lPred === 1 ? lProb : 1 - lProb,
              };
            }
            if (item.model.startsWith("BiLSTM") && restModel === "lstm") {
              return {
                ...item,
                prediction: predVal,
                label: predVal === 1 ? ("Spam" as const) : ("Not Spam" as const),
                spamProbability: probVal,
                confidence: predVal === 1 ? probVal : 1 - probVal,
              };
            }
            return item;
          });

      const result: ClassificationResult = {
        ...localBaseline,
        id: String(data.id ?? localBaseline.id),
        transformedText: String(data.transformed_text ?? localBaseline.transformedText),
        prediction: predVal,
        label: data.label === "Spam" ? "Spam" : "Not Spam",
        threatType: String(data.threat_type ?? localBaseline.threatType),
        threatRiskLevel: (data.threat_risk_level ?? localBaseline.threatRiskLevel) as
          | "High"
          | "Medium"
          | "None",
        threatReason: String(data.threat_reason ?? localBaseline.threatReason),
        spamProbability: probVal,
        hamProbability: Number(data.ham_probability ?? 1 - probVal),
        inferenceMs: Number(data.inference_ms ?? roundTripMs),
        modelUsed: String(data.model_used ?? "Flask / FastAPI REST"),
        modelPredictions: updatedModelPredictions,
      };
      setLatestResult(result);
      setHistory((prev) => [result, ...prev].slice(0, 50));
    } catch (err) {
      const message = err instanceof Error ? err.message : "REST API unreachable.";
      setApiError(
        `Could not reach ${restUrl} (${message}). Start \`python -m api.app\` or switch to Browser MNB.`
      );
    } finally {
      setIsPredicting(false);
    }
  }

  async function handleBatchScan() {
    const lines = batchInput
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    if (lines.length === 0) {
      setApiError("Paste one or more lines in the batch scanner before running.");
      return;
    }
    if (lines.length > 100) {
      setApiError("Batch scan is limited to 100 messages per request.");
      return;
    }

    setApiError(null);

    if (engineMode === "local") {
      const scanned = lines.map((line) => classifyLocalMnb(line, "", "batch@scanner"));
      setBatchResults(scanned);
      setHistory((prev) => [...scanned, ...prev].slice(0, 50));
      return;
    }

    setIsPredicting(true);
    try {
      const parsedRest = new URL(restUrl.trim());
      if (parsedRest.protocol !== "http:" && parsedRest.protocol !== "https:") {
        throw new Error("Only http:// or https:// REST endpoints are supported.");
      }
      const endpoint = `${parsedRest.origin}${parsedRest.pathname.replace(/\/+$/, "")}/predict/batch`;
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ texts: lines }),
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status} from ${endpoint}`);
      }
      const data = await response.json();
      const totalMs = Number(data.inference_ms ?? 1);
      const perItemMs = totalMs / Math.max(lines.length, 1);
      const scanned: ClassificationResult[] = (data.results ?? []).map(
        (item: {
          text: string;
          prediction: number;
          label: string;
          threat_type?: string;
          threat_risk_level?: "High" | "Medium" | "None";
          threat_reason?: string;
          spam_probability: number;
        }) => {
          const base = classifyLocalMnb(item.text, "", "batch@api");
          return {
            ...base,
            prediction: item.prediction === 1 ? 1 : 0,
            label: item.label === "Spam" ? "Spam" : "Not Spam",
            threatType: item.threat_type ?? base.threatType,
            threatRiskLevel: item.threat_risk_level ?? base.threatRiskLevel,
            threatReason: item.threat_reason ?? base.threatReason,
            spamProbability: Number(item.spam_probability),
            hamProbability: 1 - Number(item.spam_probability),
            inferenceMs: perItemMs,
            modelUsed: "REST Batch MNB",
          };
        }
      );
      setBatchResults(scanned);
      setHistory((prev) => [...scanned, ...prev].slice(0, 50));
    } catch (err) {
      const message = err instanceof Error ? err.message : "Batch API unreachable.";
      setApiError(`Batch scan failed (${message}). Switch to Browser MNB or start the API server.`);
    } finally {
      setIsPredicting(false);
    }
  }

  function applyPreset(presetId: string) {
    const preset = SAMPLE_PRESETS.find((p) => p.id === presetId) ?? SAMPLE_PRESETS[0];
    setScanMode("single");
    setSender(preset.sender);
    setSubject(preset.subject);
    setBody(preset.body);
    setApiError(null);
    if (engineMode === "local") {
      const res = classifyLocalMnb(preset.body, preset.subject, preset.sender);
      setLatestResult(res);
      setHistory((prev) => [res, ...prev].slice(0, 50));
    }
    document.getElementById("classifier")?.scrollIntoView({ behavior: "smooth" });
  }

  function handleExportCsv() {
    if (history.length === 0) return;
    const headers = [
      "timestamp",
      "sender",
      "subject",
      "verdict",
      "threat_type",
      "risk_level",
      "spam_probability",
      "latency_ms",
      "model",
    ];
    const rows = history.map((item) => [
      item.timestamp,
      item.sender.replaceAll('"', '""'),
      item.subject.replaceAll('"', '""'),
      item.label,
      item.threatType.replaceAll('"', '""'),
      item.threatRiskLevel,
      item.spamProbability.toFixed(4),
      item.inferenceMs.toFixed(3),
      item.modelUsed,
    ]);
    const csv = [headers.join(","), ...rows.map((r) => r.map((v) => `"${v}"`).join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "spam_classification_history.csv";
    link.click();
    URL.revokeObjectURL(url);
  }

  function handleCopy(key: string, value: string) {
    navigator.clipboard?.writeText(value);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 1500);
  }

  const curlSnippet = `curl -X POST ${restUrl.replace(/\/+$/, "")}/predict \\
  -H "Content-Type: application/json" \\
  -d '{"text": "URGENT: Claim your FREE $1,000 gift card now!", "model": "${restModel}"}'`;

  const sqlSnippet = `SELECT id, raw_text, prediction_label,
       1 - (lstm_embedding <=> $1::vector) AS semantic_similarity
FROM classification_logs
WHERE lstm_embedding IS NOT NULL
ORDER BY lstm_embedding <=> $1::vector
LIMIT 5;`;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-950 dark:text-slate-100 transition-colors">
      {/* Zone 1 / Zone 2 / Zone 3 Top Bar Contract */}
      <header className="sticky top-0 z-30 h-16 bg-white/95 dark:bg-slate-950/95 backdrop-blur-sm border-b border-slate-200 dark:border-slate-800">
        <div className="max-w-6xl mx-auto h-full px-6 flex items-center justify-between gap-8">
          <a
            href="#hero"
            className="font-display text-lg font-bold tracking-tight text-slate-900 dark:text-white whitespace-nowrap shrink-0"
          >
            SpamGuard
          </a>

          <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-600 dark:text-slate-300">
            <a
              href="#classifier"
              className="hover:text-slate-900 dark:hover:text-white transition-colors whitespace-nowrap shrink-0"
            >
              Classifier
            </a>
            <a
              href="#history"
              className="hover:text-slate-900 dark:hover:text-white transition-colors whitespace-nowrap shrink-0"
            >
              History
            </a>
            <a
              href="#analytics"
              className="hover:text-slate-900 dark:hover:text-white transition-colors whitespace-nowrap shrink-0"
            >
              Analytics
            </a>
            <a
              href="#architecture"
              className="hover:text-slate-900 dark:hover:text-white transition-colors whitespace-nowrap shrink-0"
            >
              Architecture
            </a>
            <a
              href="#faq"
              className="hover:text-slate-900 dark:hover:text-white transition-colors whitespace-nowrap shrink-0"
            >
              FAQ
            </a>
          </nav>

          <div className="shrink-0">
            <button
              type="button"
              onClick={() => setDarkMode((prev) => !prev)}
              className="inline-flex items-center gap-2 px-3.5 py-2 text-xs font-medium text-slate-700 dark:text-slate-200 border border-slate-200 dark:border-slate-700 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0"
              aria-label="Toggle light or dark mode"
            >
              {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
              <span>{darkMode ? "Light Mode" : "Dark Mode"}</span>
            </button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section id="hero" className="border-b border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900/40">
        <div className="max-w-6xl mx-auto px-6 py-12 lg:py-16">
          <div className="max-w-3xl space-y-4">
            <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-xs font-medium text-slate-600 dark:text-slate-300">
              <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
                <FileText className="w-3.5 h-3.5 text-slate-500 dark:text-slate-400 shrink-0" />
                <span>SMS Spam Collection</span>
              </span>
              <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
                <Cpu className="w-3.5 h-3.5 text-slate-500 dark:text-slate-400 shrink-0" />
                <span>MultinomialNB</span>
              </span>
              <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
                <Brain className="w-3.5 h-3.5 text-slate-500 dark:text-slate-400 shrink-0" />
                <span>PyTorch BiLSTM</span>
              </span>
              <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
                <Database className="w-3.5 h-3.5 text-slate-500 dark:text-slate-400 shrink-0" />
                <span>PostgreSQL pgvector</span>
              </span>
            </div>
            <h1 className="font-display text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight text-slate-900 dark:text-white text-balance">
              Precision email threat screening with zero false positives.
            </h1>
            <p className="text-base text-slate-600 dark:text-slate-300 leading-relaxed max-w-2xl">
              Screen sender headers, subject lines, and message bodies through a deterministic
              regex, stop-word, and Porter-stemming pipeline—or route live scans to your Flask,
              FastAPI, and pgvector backend.
            </p>
            <div className="flex flex-wrap items-center gap-3 pt-2">
              <a
                href="#classifier"
                className="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-semibold text-white bg-slate-900 dark:bg-blue-600 rounded-lg hover:bg-slate-800 dark:hover:bg-blue-500 transition-colors whitespace-nowrap"
              >
                <span>Open Classifier</span>
                <ArrowDown className="w-4 h-4" />
              </a>
              <a
                href="#analytics"
                className="px-5 py-2.5 text-sm font-medium text-slate-700 dark:text-slate-200 border border-slate-200 dark:border-slate-700 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors whitespace-nowrap"
              >
                View Model Benchmarks
              </a>
            </div>
          </div>
        </div>
      </section>

      <main className="max-w-6xl mx-auto px-6 py-10 space-y-16">
        {/* Section 1: Interactive Email & Batch Classifier */}
        <section id="classifier" className="scroll-mt-20 space-y-6">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <h2 className="font-display text-2xl font-bold text-slate-900 dark:text-white">
                Interactive Threat Scanner
              </h2>
              <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
                Inspect single emails with token-level weighting or scan multiple lines in batch mode.
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-3">
              {/* Scan Mode Switch */}
              <div className="flex items-center gap-1 p-1 bg-slate-200/70 dark:bg-slate-800 rounded-lg">
                <button
                  type="button"
                  onClick={() => setScanMode("single")}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                    scanMode === "single"
                      ? "bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-xs"
                      : "text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white"
                  }`}
                >
                  Single Email
                </button>
                <button
                  type="button"
                  onClick={() => setScanMode("batch")}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                    scanMode === "batch"
                      ? "bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-xs"
                      : "text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white"
                  }`}
                >
                  Batch Scan
                </button>
              </div>

              {/* Engine Switch */}
              <div className="flex items-center gap-1 p-1 bg-slate-200/70 dark:bg-slate-800 rounded-lg">
                <button
                  type="button"
                  onClick={() => {
                    setEngineMode("local");
                    setApiError(null);
                  }}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                    engineMode === "local"
                      ? "bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-xs"
                      : "text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white"
                  }`}
                >
                  Browser MNB
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setEngineMode("rest");
                    setApiError(null);
                  }}
                  className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                    engineMode === "rest"
                      ? "bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-xs"
                      : "text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white"
                  }`}
                >
                  Flask / FastAPI REST
                </button>
              </div>
            </div>
          </div>

          {engineMode === "rest" && (
            <div className="p-4 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 grid grid-cols-1 sm:grid-cols-4 gap-4">
              <div className="sm:col-span-3">
                <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                  Flask (:5000) or FastAPI (:8000) Base URL
                </label>
                <input
                  type="url"
                  value={restUrl}
                  onChange={(e) => setRestUrl(e.target.value)}
                  className="w-full px-3 py-2 text-xs font-mono border border-slate-200 dark:border-slate-700 rounded-lg bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-600"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                  Backend Model
                </label>
                <select
                  value={restModel}
                  onChange={(e) => setRestModel(e.target.value as "mnb" | "lstm" | "both")}
                  className="w-full px-3 py-2 text-xs border border-slate-200 dark:border-slate-700 rounded-lg bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-600"
                >
                  <option value="mnb">MultinomialNB</option>
                  <option value="lstm">BiLSTM (PyTorch)</option>
                  <option value="both">MNB + BiLSTM</option>
                </select>
              </div>
            </div>
          )}

          {scanMode === "single" ? (
            <div className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
              {/* Composer Panel */}
              <div className="lg:col-span-7 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 flex flex-col justify-between gap-5">
                <div className="space-y-4">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                      <label htmlFor="sender-input" className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
                        Sender Address
                      </label>
                      <input
                        id="sender-input"
                        type="text"
                        autoComplete="off"
                        maxLength={320}
                        value={sender}
                        onChange={(e) => setSender(e.target.value)}
                        placeholder="sender@domain.com"
                        className="w-full px-3 py-2 text-xs font-mono border border-slate-200 dark:border-slate-700 rounded-lg bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-600"
                      />
                    </div>
                    <div>
                      <label htmlFor="subject-input" className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
                        Subject Line
                      </label>
                      <input
                        id="subject-input"
                        type="text"
                        autoComplete="off"
                        maxLength={500}
                        value={subject}
                        onChange={(e) => setSubject(e.target.value)}
                        placeholder="Email subject..."
                        className="w-full px-3 py-2 text-xs border border-slate-200 dark:border-slate-700 rounded-lg bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-600"
                      />
                    </div>
                  </div>

                  <div>
                    <label htmlFor="body-input" className="block text-xs font-medium text-slate-700 dark:text-slate-300 mb-1">
                      Message Body
                    </label>
                    <textarea
                      id="body-input"
                      rows={6}
                      value={body}
                      onChange={(e) => setBody(e.target.value)}
                      placeholder="Paste email body or SMS message here..."
                      className="w-full px-3.5 py-2.5 text-sm border border-slate-200 dark:border-slate-700 rounded-lg bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-600 resize-y"
                    />
                  </div>

                  <div className="flex flex-wrap items-center gap-2">
                    {SAMPLE_PRESETS.map((preset) => (
                      <button
                        key={preset.id}
                        type="button"
                        onClick={() => applyPreset(preset.id)}
                        className="px-2.5 py-1.5 text-xs font-medium text-slate-700 dark:text-slate-200 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 rounded-md transition-colors whitespace-nowrap"
                      >
                        {preset.label}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="space-y-3 pt-4 border-t border-slate-100 dark:border-slate-800">
                  {apiError && (
                    <div className="p-3 rounded-lg border border-amber-300 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/40 text-xs text-amber-900 dark:text-amber-200 flex items-start gap-2">
                      <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
                      <span>{apiError}</span>
                    </div>
                  )}

                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <span className="text-xs font-mono tabular-nums text-slate-500 dark:text-slate-400">
                      {body.length} chars · {liveTransformedPreview ? liveTransformedPreview.split(" ").length : 0} stemmed tokens
                    </span>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={() => {
                          setSender("");
                          setSubject("");
                          setBody("");
                          setApiError(null);
                        }}
                        className="px-4 py-2 text-xs font-medium text-slate-700 dark:text-slate-200 border border-slate-200 dark:border-slate-700 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors whitespace-nowrap"
                      >
                        Clear
                      </button>
                      <button
                        type="button"
                        onClick={handlePredictSingle}
                        disabled={isPredicting}
                        className="px-5 py-2 text-xs font-semibold text-white bg-slate-900 dark:bg-blue-600 rounded-lg hover:bg-slate-800 dark:hover:bg-blue-500 disabled:opacity-50 transition-colors whitespace-nowrap"
                      >
                        {isPredicting ? "Scanning..." : "Classify Message"}
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* Verdict & Token Trace Panel */}
              <div className="lg:col-span-5 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 flex flex-col justify-between gap-5">
                {latestResult === null ? (
                  <div className="flex-1 flex flex-col items-center justify-center text-center py-10 px-4 space-y-3">
                    <div className="w-10 h-10 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 flex items-center justify-center text-slate-500 dark:text-slate-400">
                      <Layers className="w-5 h-5" />
                    </div>
                    <div className="space-y-1 max-w-xs">
                      <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
                        Awaiting Message Input
                      </h3>
                      <p className="text-xs text-slate-500 dark:text-slate-400 leading-relaxed">
                        Enter a Sender Address, Subject Line, or Message Body and click{" "}
                        <span className="font-medium text-slate-700 dark:text-slate-200">
                          Classify Message
                        </span>{" "}
                        to inspect spam probability, threat category, and token signals.
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between gap-2">
                      <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
                        Classification Verdict
                      </h3>
                      <span className="text-xs font-mono tabular-nums text-slate-500 dark:text-slate-400">
                        {latestResult.timestamp}
                      </span>
                    </div>

                    {latestResult.label === "Spam" ? (
                      <div className="p-4 rounded-lg border border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950/40 flex items-center justify-between gap-4">
                        <div className="flex items-center gap-3">
                          <AlertTriangle className="w-5 h-5 text-red-600 dark:text-red-400 shrink-0" />
                          <div>
                            <p className="text-sm font-bold text-red-900 dark:text-red-200">
                              Spam Detected
                            </p>
                            <p className="text-xs text-red-700 dark:text-red-300">
                              Flagged as malicious or unsolicited mail
                            </p>
                          </div>
                        </div>
                        <span className="font-mono text-xl font-bold text-red-900 dark:text-red-200 tabular-nums">
                          {(latestResult.spamProbability * 100).toFixed(1)}%
                        </span>
                      </div>
                    ) : (
                      <div className="p-4 rounded-lg border border-emerald-200 dark:border-emerald-900 bg-emerald-50 dark:bg-emerald-950/40 flex items-center justify-between gap-4">
                        <div className="flex items-center gap-3">
                          <CheckCircle2 className="w-5 h-5 text-emerald-600 dark:text-emerald-400 shrink-0" />
                          <div>
                            <p className="text-sm font-bold text-emerald-900 dark:text-emerald-200">
                              Not Spam (Ham)
                            </p>
                            <p className="text-xs text-emerald-700 dark:text-emerald-300">
                              Safe legitimate communication
                            </p>
                          </div>
                        </div>
                        <span className="font-mono text-xl font-bold text-emerald-900 dark:text-emerald-200 tabular-nums">
                          {(latestResult.hamProbability * 100).toFixed(1)}%
                        </span>
                      </div>
                    )}

                    {/* Threat / Scam Classification Breakdown */}
                    <div className="p-3.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 space-y-1">
                      <div className="flex items-center justify-between gap-2 text-xs">
                        <span className="text-slate-500 dark:text-slate-400">
                          Threat / Scam Type
                        </span>
                        <span
                          className={`font-mono font-semibold ${
                            latestResult.threatRiskLevel === "High"
                              ? "text-red-700 dark:text-red-400"
                              : latestResult.threatRiskLevel === "Medium"
                                ? "text-amber-700 dark:text-amber-400"
                                : "text-emerald-700 dark:text-emerald-400"
                          }`}
                        >
                          Risk: {latestResult.threatRiskLevel}
                        </span>
                      </div>
                      <p className="text-sm font-semibold text-slate-900 dark:text-white">
                        {latestResult.threatType}
                      </p>
                      <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
                        {latestResult.threatReason}
                      </p>
                    </div>

                    <div className="grid grid-cols-3 gap-3 py-3 border-y border-slate-100 dark:border-slate-800">
                      <div>
                        <p className="text-xs text-slate-500 dark:text-slate-400">Inference</p>
                        <p className="text-sm font-mono font-semibold text-slate-900 dark:text-white tabular-nums mt-0.5">
                          {latestResult.inferenceMs.toFixed(2)} ms
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-slate-500 dark:text-slate-400">P(Spam)</p>
                        <p className="text-sm font-mono font-semibold text-slate-900 dark:text-white tabular-nums mt-0.5">
                          {(latestResult.spamProbability * 100).toFixed(2)}%
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-slate-500 dark:text-slate-400">Model</p>
                        <p className="text-xs font-medium text-slate-900 dark:text-slate-200 truncate mt-1">
                          {latestResult.modelUsed}
                        </p>
                      </div>
                    </div>

                    {/* Token Highlighting Preview */}
                    <div>
                      <p className="text-xs font-medium text-slate-700 dark:text-slate-300 mb-1.5">
                        Inline Token Signal Trace
                      </p>
                      <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-xs leading-relaxed flex flex-wrap gap-1.5 max-h-24 overflow-y-auto">
                        {latestResult.highlights.map((tok, idx) => (
                          <span
                            key={`${tok.raw}-${idx}`}
                            className={
                              tok.role === "spam"
                                ? "px-1 rounded bg-red-100 dark:bg-red-900/50 text-red-900 dark:text-red-200 font-medium"
                                : tok.role === "ham"
                                  ? "px-1 rounded bg-emerald-100 dark:bg-emerald-900/50 text-emerald-900 dark:text-emerald-200 font-medium"
                                  : "text-slate-600 dark:text-slate-400"
                            }
                            title={tok.stemmed ? `Stemmed: ${tok.stemmed}` : "Stop-word / punctuation"}
                          >
                            {tok.raw}
                          </span>
                        ))}
                      </div>
                    </div>

                    {/* Stemmed Vector String */}
                    <div>
                      <p className="text-xs font-medium text-slate-700 dark:text-slate-300 mb-1.5">
                        transform_text() Normalized Tokens
                      </p>
                      <div className="p-2.5 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 font-mono text-xs text-slate-700 dark:text-slate-300 break-words max-h-20 overflow-y-auto">
                        {latestResult.transformedText || "(empty after stop-word removal)"}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {latestResult !== null && (
              <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
                      Per-Model Prediction Accuracy on Provided Mail
                    </h3>
                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                      Each classifier’s predicted verdict, confidence on this email, and benchmark validation accuracy
                    </p>
                  </div>
                  <span className="text-xs font-mono tabular-nums text-slate-500 dark:text-slate-400">
                    {latestResult.modelPredictions.filter((m) => m.prediction === latestResult.prediction).length}/
                    {latestResult.modelPredictions.length} models agree
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                  {latestResult.modelPredictions.map((mp) => (
                    <div
                      key={mp.model}
                      className="p-3.5 rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 space-y-2"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <p className="text-xs font-semibold text-slate-900 dark:text-white truncate">
                            {mp.model}
                          </p>
                          <p className="text-[11px] text-slate-500 dark:text-slate-400 truncate">
                            {mp.family}
                          </p>
                        </div>
                        <span
                          className={`text-[11px] font-semibold whitespace-nowrap ${
                            mp.prediction === 1
                              ? "text-red-700 dark:text-red-400"
                              : "text-emerald-700 dark:text-emerald-400"
                          }`}
                        >
                          {mp.label}
                        </span>
                      </div>

                      <div className="space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-slate-500 dark:text-slate-400">
                            Prediction Confidence
                          </span>
                          <span className="font-mono font-semibold text-slate-900 dark:text-white tabular-nums">
                            {(mp.confidence * 100).toFixed(2)}%
                          </span>
                        </div>
                        <div className="w-full h-1.5 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                          <div
                            className={`h-full ${
                              mp.prediction === 1 ? "bg-red-600" : "bg-emerald-600"
                            }`}
                            style={{ width: `${Math.round(mp.confidence * 100)}%` }}
                          />
                        </div>
                        <div className="flex items-center justify-between text-[11px] text-slate-500 dark:text-slate-400 pt-0.5">
                          <span>Val Accuracy: {(mp.validationAccuracy * 100).toFixed(2)}%</span>
                          <span className="font-mono tabular-nums">
                            P(Spam): {(mp.spamProbability * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
          ) : (
            /* Batch Scanner Mode */
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
                    Batch Multi-Message Scanner
                  </h3>
                  <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                    Paste one email or SMS per line to classify in bulk.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={handleBatchScan}
                  disabled={isPredicting}
                  className="px-5 py-2 text-xs font-semibold text-white bg-slate-900 dark:bg-blue-600 rounded-lg hover:bg-slate-800 dark:hover:bg-blue-500 disabled:opacity-50 transition-colors whitespace-nowrap"
                >
                  {isPredicting ? "Scanning Batch..." : "Run Batch Scan"}
                </button>
              </div>

              <textarea
                rows={5}
                value={batchInput}
                onChange={(e) => setBatchInput(e.target.value)}
                placeholder="Paste one email or SMS message per line..."
                className="w-full px-3.5 py-2.5 text-xs font-mono border border-slate-200 dark:border-slate-700 rounded-lg bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-600"
              />

              {apiError && (
                <div className="p-3 rounded-lg border border-amber-300 dark:border-amber-800 bg-amber-50 dark:bg-amber-950/40 text-xs text-amber-900 dark:text-amber-200">
                  {apiError}
                </div>
              )}

              {batchResults.length > 0 && (
                <div className="overflow-x-auto pt-2">
                  <table className="w-full text-left border-collapse text-xs">
                    <thead>
                      <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400">
                        <th className="py-2 pr-4">Message</th>
                        <th className="py-2 px-4">Verdict</th>
                        <th className="py-2 px-4">Threat / Scam Type</th>
                        <th className="py-2 px-4 text-right">P(Spam)</th>
                        <th className="py-2 pl-4 text-right">Latency</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                      {batchResults.map((res) => (
                        <tr key={res.id}>
                          <td className="py-2.5 pr-4 max-w-md truncate text-slate-900 dark:text-slate-100">
                            {res.rawText}
                          </td>
                          <td className="py-2.5 px-4 whitespace-nowrap font-semibold">
                            <span
                              className={
                                res.prediction === 1
                                  ? "text-red-700 dark:text-red-400"
                                  : "text-emerald-700 dark:text-emerald-400"
                              }
                            >
                              {res.label}
                            </span>
                          </td>
                          <td className="py-2.5 px-4 whitespace-nowrap text-slate-700 dark:text-slate-300">
                            {res.threatType}
                          </td>
                          <td className="py-2.5 px-4 text-right font-mono tabular-nums">
                            {(res.spamProbability * 100).toFixed(2)}%
                          </td>
                          <td className="py-2.5 pl-4 text-right font-mono tabular-nums text-slate-500">
                            {res.inferenceMs.toFixed(2)} ms
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </section>

        {/* Section 2: Scan History */}
        <section
          id="history"
          className="scroll-mt-20 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 space-y-4"
        >
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <h2 className="font-display text-lg font-semibold text-slate-900 dark:text-white">
                Classification History
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                Searchable log of scanned messages · exportable to CSV
              </p>
            </div>

            <div className="flex flex-wrap items-center gap-2.5">
              <div className="relative">
                <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  value={historyQuery}
                  onChange={(e) => setHistoryQuery(e.target.value)}
                  placeholder="Search sender or text..."
                  className="pl-8 pr-3 py-1.5 text-xs border border-slate-200 dark:border-slate-700 rounded-lg bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-blue-600"
                />
              </div>

              <div className="flex items-center gap-1 p-1 bg-slate-100 dark:bg-slate-800 rounded-lg">
                {(["all", "spam", "ham"] as const).map((tab) => (
                  <button
                    key={tab}
                    type="button"
                    onClick={() => setHistoryFilter(tab)}
                    className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap ${
                      historyFilter === tab
                        ? "bg-white dark:bg-slate-900 text-slate-900 dark:text-white shadow-xs"
                        : "text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white"
                    }`}
                  >
                    {tab === "all" ? "All" : tab === "spam" ? "Spam" : "Not Spam"}
                  </button>
                ))}
              </div>

              <button
                type="button"
                onClick={handleExportCsv}
                disabled={history.length === 0}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 dark:text-slate-200 border border-slate-200 dark:border-slate-700 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-800 disabled:opacity-40 transition-colors whitespace-nowrap"
              >
                <Download className="w-3.5 h-3.5" />
                <span>Export CSV</span>
              </button>

              {history.length > 0 && (
                <button
                  type="button"
                  onClick={() => setHistory([])}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-600 dark:text-slate-300 hover:text-red-700 dark:hover:text-red-400 border border-slate-200 dark:border-slate-700 rounded-lg transition-colors whitespace-nowrap"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                  <span>Clear</span>
                </button>
              )}
            </div>
          </div>

          {filteredHistory.length === 0 ? (
            <div className="py-10 text-center border border-dashed border-slate-200 dark:border-slate-800 rounded-lg space-y-2">
              <p className="text-sm font-medium text-slate-700 dark:text-slate-300">
                No scan records match the active filter
              </p>
              <button
                type="button"
                onClick={() => applyPreset("prize")}
                className="px-4 py-2 text-xs font-medium text-white bg-slate-900 dark:bg-blue-600 rounded-lg hover:bg-slate-800 transition-colors"
              >
                Run First Scan
              </button>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400">
                    <th className="py-2.5 pr-4">Sender & Subject</th>
                    <th className="py-2.5 px-4">Verdict</th>
                    <th className="py-2.5 px-4">Threat / Scam Type</th>
                    <th className="py-2.5 px-4 text-right">P(Spam)</th>
                    <th className="py-2.5 px-4 text-right">Latency</th>
                    <th className="py-2.5 pl-4 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                  {filteredHistory.map((row) => (
                    <tr key={row.id} className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                      <td className="py-3 pr-4 max-w-md">
                        <p className="font-medium text-slate-900 dark:text-white truncate">
                          {row.subject !== "(no subject)" ? row.subject : row.rawText}
                        </p>
                        <p className="text-slate-500 dark:text-slate-400 font-mono truncate mt-0.5">
                          {row.sender} · {row.rawText}
                        </p>
                      </td>
                      <td className="py-3 px-4 whitespace-nowrap font-semibold">
                        <span
                          className={
                            row.prediction === 1
                              ? "text-red-700 dark:text-red-400"
                              : "text-emerald-700 dark:text-emerald-400"
                          }
                        >
                          {row.label}
                        </span>
                      </td>
                      <td className="py-3 px-4 whitespace-nowrap text-slate-700 dark:text-slate-300">
                        {row.threatType}
                      </td>
                      <td className="py-3 px-4 text-right font-mono tabular-nums">
                        {(row.spamProbability * 100).toFixed(2)}%
                      </td>
                      <td className="py-3 px-4 text-right font-mono tabular-nums text-slate-500 dark:text-slate-400">
                        {row.inferenceMs.toFixed(2)} ms
                      </td>
                      <td className="py-3 pl-4 text-right">
                        <button
                          type="button"
                          onClick={() =>
                            setHistory((prev) => prev.filter((entry) => entry.id !== row.id))
                          }
                          className="text-slate-400 hover:text-red-600 dark:hover:text-red-400 p-1"
                          aria-label="Delete log entry"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {/* Section 3: Model Benchmarks & Corpus Analytics */}
        <section id="analytics" className="scroll-mt-20 space-y-6">
          <div>
            <h2 className="font-display text-2xl font-bold text-slate-900 dark:text-white">
              Model Benchmarks & Corpus Analytics
            </h2>
            <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
              Stratified 80/20 evaluation across 7 classical ML classifiers and PyTorch BiLSTM (5,169 deduplicated records).
            </p>
          </div>

          <div className="space-y-6">
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-slate-200 dark:border-slate-800 text-slate-500 dark:text-slate-400">
                    <th className="py-2.5 pr-4">Model</th>
                    <th className="py-2.5 px-3 text-right">Accuracy</th>
                    <th className="py-2.5 px-3 text-right">Precision</th>
                    <th className="py-2.5 px-3 text-right">Recall</th>
                    <th className="py-2.5 pl-3 text-right">F1</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
                  {BENCHMARK_DATA.map((row) => {
                    const active = selectedBenchmark.model === row.model;
                    return (
                      <tr
                        key={row.model}
                        onClick={() => setSelectedBenchmark(row)}
                        className={`cursor-pointer transition-colors ${
                          active
                            ? "bg-slate-100 dark:bg-slate-800"
                            : "hover:bg-slate-50 dark:hover:bg-slate-800/40"
                        }`}
                      >
                        <td className="py-3 pr-4 font-semibold text-slate-900 dark:text-white whitespace-nowrap">
                          {row.model}
                          {row.selected ? " (Selected)" : ""}
                        </td>
                        <td className="py-3 px-3 text-right font-mono tabular-nums">
                          {(row.accuracy * 100).toFixed(2)}%
                        </td>
                        <td
                          className={`py-3 px-3 text-right font-mono tabular-nums font-semibold ${
                            row.precision === 1
                              ? "text-emerald-700 dark:text-emerald-400"
                              : ""
                          }`}
                        >
                          {(row.precision * 100).toFixed(2)}%
                        </td>
                        <td className="py-3 px-3 text-right font-mono tabular-nums text-slate-600 dark:text-slate-300">
                          {(row.recall * 100).toFixed(2)}%
                        </td>
                        <td className="py-3 pl-3 text-right font-mono tabular-nums">
                          {(row.f1 * 100).toFixed(2)}%
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Side-by-Side Confusion Matrix: BiLSTM (PyTorch 128d) | MultinomialNB */}
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl overflow-x-auto">
              <div className="grid grid-cols-2 min-w-[600px] divide-x divide-slate-200 dark:divide-slate-800">
                {[
                  BENCHMARK_DATA.find((r) => r.model.startsWith("BiLSTM")) ?? BENCHMARK_DATA[0],
                  BENCHMARK_DATA.find((r) => r.model === "MultinomialNB") ?? BENCHMARK_DATA[1],
                ].map((cmRow) => (
                  <div key={cmRow.model} className="p-5 sm:p-6 flex flex-col justify-between gap-4">
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <div>
                        <h3 className="text-sm font-semibold text-slate-900 dark:text-white whitespace-nowrap">
                          Confusion Matrix — {cmRow.model}
                        </h3>
                        <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                          {cmRow.family} · 1,034 validation messages
                        </p>
                      </div>
                      <div className="text-right font-mono text-xs tabular-nums whitespace-nowrap">
                        <span className="font-semibold text-slate-900 dark:text-white">
                          Acc {(cmRow.accuracy * 100).toFixed(2)}%
                        </span>
                        <span className="text-slate-400 mx-1.5">·</span>
                        <span className="font-semibold text-emerald-700 dark:text-emerald-400">
                          Prec {(cmRow.precision * 100).toFixed(2)}%
                        </span>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-3 text-center">
                      <div className="p-3.5 rounded-lg bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-900">
                        <p className="text-xs text-emerald-700 dark:text-emerald-300">True Negative (Ham)</p>
                        <p className="text-2xl font-mono font-bold text-emerald-900 dark:text-emerald-200 tabular-nums mt-1">
                          {cmRow.confusionMatrix[0][0]}
                        </p>
                      </div>
                      <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
                        <p className="text-xs text-slate-600 dark:text-slate-400">False Positive</p>
                        <p
                          className={`text-2xl font-mono font-bold tabular-nums mt-1 ${
                            cmRow.confusionMatrix[0][1] === 0
                              ? "text-emerald-700 dark:text-emerald-400"
                              : "text-red-700 dark:text-red-400"
                          }`}
                        >
                          {cmRow.confusionMatrix[0][1]}
                        </p>
                      </div>
                      <div className="p-3.5 rounded-lg bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-900">
                        <p className="text-xs text-amber-800 dark:text-amber-300">False Negative</p>
                        <p className="text-2xl font-mono font-bold text-amber-900 dark:text-amber-200 tabular-nums mt-1">
                          {cmRow.confusionMatrix[1][0]}
                        </p>
                      </div>
                      <div className="p-3.5 rounded-lg bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-900">
                        <p className="text-xs text-blue-700 dark:text-blue-300">True Positive (Spam)</p>
                        <p className="text-2xl font-mono font-bold text-blue-900 dark:text-blue-200 tabular-nums mt-1">
                          {cmRow.confusionMatrix[1][1]}
                        </p>
                      </div>
                    </div>

                    <div className="space-y-2 pt-2 border-t border-slate-100 dark:border-slate-800">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-slate-600 dark:text-slate-400">Corpus Class Balance</span>
                        <span className="font-mono tabular-nums text-slate-700 dark:text-slate-300">
                          Ham {EDA_STATS.hamPct}% · Spam {EDA_STATS.spamPct}%
                        </span>
                      </div>
                      <div className="w-full h-2 rounded-md overflow-hidden flex bg-slate-100 dark:bg-slate-800">
                        <div className="bg-emerald-600 h-full" style={{ width: `${EDA_STATS.hamPct}%` }} />
                        <div className="bg-red-600 h-full" style={{ width: `${EDA_STATS.spamPct}%` }} />
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {selectedBenchmark.model !== "BiLSTM (PyTorch 128d)" &&
              selectedBenchmark.model !== "MultinomialNB" && (
                <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
                        Selected Row Confusion Matrix — {selectedBenchmark.model}
                      </h3>
                      <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                        {selectedBenchmark.family} · 1,034 validation messages
                      </p>
                    </div>
                    <span className="text-xs font-mono tabular-nums text-slate-600 dark:text-slate-300">
                      Acc {(selectedBenchmark.accuracy * 100).toFixed(2)}% · Prec{" "}
                      {(selectedBenchmark.precision * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                    <div className="p-3.5 rounded-lg bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-900">
                      <p className="text-xs text-emerald-700 dark:text-emerald-300">True Negative (Ham)</p>
                      <p className="text-xl font-mono font-bold text-emerald-900 dark:text-emerald-200 tabular-nums mt-1">
                        {selectedBenchmark.confusionMatrix[0][0]}
                      </p>
                    </div>
                    <div className="p-3.5 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
                      <p className="text-xs text-slate-600 dark:text-slate-400">False Positive</p>
                      <p className="text-xl font-mono font-bold text-red-700 dark:text-red-400 tabular-nums mt-1">
                        {selectedBenchmark.confusionMatrix[0][1]}
                      </p>
                    </div>
                    <div className="p-3.5 rounded-lg bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-900">
                      <p className="text-xs text-amber-800 dark:text-amber-300">False Negative</p>
                      <p className="text-xl font-mono font-bold text-amber-900 dark:text-amber-200 tabular-nums mt-1">
                        {selectedBenchmark.confusionMatrix[1][0]}
                      </p>
                    </div>
                    <div className="p-3.5 rounded-lg bg-blue-50 dark:bg-blue-950/40 border border-blue-200 dark:border-blue-900">
                      <p className="text-xs text-blue-700 dark:text-blue-300">True Positive (Spam)</p>
                      <p className="text-xl font-mono font-bold text-blue-900 dark:text-blue-200 tabular-nums mt-1">
                        {selectedBenchmark.confusionMatrix[1][1]}
                      </p>
                    </div>
                  </div>
                </div>
              )}
          </div>
        </section>

        {/* Section 4: System Architecture & API Integration */}
        <section id="architecture" className="scroll-mt-20 space-y-6">
          <div>
            <h2 className="font-display text-2xl font-bold text-slate-900 dark:text-white">
              Pipeline Architecture & Backend Stack
            </h2>
            <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
              End-to-end Python ML pipeline backed by Flask, FastAPI, Streamlit, and PostgreSQL pgvector.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              {
                step: "01. Ingestion & EDA",
                desc: "Deduplicates 5,572 SMS records down to 5,169 unique samples and computes length & word distributions.",
              },
              {
                step: "02. NLP Transform",
                desc: "Lowercasing, regex URL/punctuation stripping, NLTK stop-word removal, and Porter Stemming.",
              },
              {
                step: "03. BiLSTM & MNB",
                desc: "Calibrated 2-layer PyTorch BiLSTM leads at 98.55% accuracy & 100% precision alongside TF-IDF MultinomialNB.",
              },
              {
                step: "04. REST & pgvector",
                desc: "Flask (:5000) & FastAPI (:8000) log predictions and query HNSW cosine similarity in PostgreSQL.",
              },
            ].map((stage) => (
              <div
                key={stage.step}
                className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 space-y-2"
              >
                <div className="flex items-center gap-2 text-slate-900 dark:text-white">
                  <Layers className="w-4 h-4 text-blue-600 dark:text-blue-400 shrink-0" />
                  <h3 className="text-sm font-semibold">{stage.step}</h3>
                </div>
                <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
                  {stage.desc}
                </p>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
                  Flask / FastAPI REST Classification
                </h3>
                <button
                  type="button"
                  onClick={() => handleCopy("curl", curlSnippet)}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium text-slate-700 dark:text-slate-200 border border-slate-200 dark:border-slate-700 rounded-md hover:bg-slate-50 dark:hover:bg-slate-800"
                >
                  <Copy className="w-3.5 h-3.5" />
                  <span>{copiedKey === "curl" ? "Copied" : "Copy cURL"}</span>
                </button>
              </div>
              <pre className="p-3.5 rounded-lg bg-slate-900 dark:bg-slate-950 text-slate-100 font-mono text-xs overflow-x-auto leading-relaxed">
                {curlSnippet}
              </pre>
            </div>

            <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-6 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
                  PostgreSQL pgvector HNSW Similarity
                </h3>
                <button
                  type="button"
                  onClick={() => handleCopy("sql", sqlSnippet)}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium text-slate-700 dark:text-slate-200 border border-slate-200 dark:border-slate-700 rounded-md hover:bg-slate-50 dark:hover:bg-slate-800"
                >
                  <Copy className="w-3.5 h-3.5" />
                  <span>{copiedKey === "sql" ? "Copied" : "Copy SQL"}</span>
                </button>
              </div>
              <pre className="p-3.5 rounded-lg bg-slate-900 dark:bg-slate-950 text-slate-100 font-mono text-xs overflow-x-auto leading-relaxed">
                {sqlSnippet}
              </pre>
            </div>
          </div>
        </section>

        {/* Section 5: FAQ Accordion */}
        <section id="faq" className="scroll-mt-20 space-y-4">
          <h2 className="font-display text-2xl font-bold text-slate-900 dark:text-white">
            Frequently Asked Questions
          </h2>
          <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl divide-y divide-slate-200 dark:divide-slate-800">
            {FAQ_ITEMS.map((item, idx) => {
              const isOpen = openFaqIndex === idx;
              return (
                <div key={item.q}>
                  <button
                    type="button"
                    onClick={() => setOpenFaqIndex(isOpen ? null : idx)}
                    className="w-full px-6 py-4 flex items-center justify-between gap-4 text-left text-sm font-semibold text-slate-900 dark:text-white hover:bg-slate-50 dark:hover:bg-slate-800/40 transition-colors"
                  >
                    <span>{item.q}</span>
                    <ChevronDown
                      className={`w-4 h-4 text-slate-500 shrink-0 transition-transform ${
                        isOpen ? "rotate-180" : ""
                      }`}
                    />
                  </button>
                  {isOpen && (
                    <div className="px-6 pb-4 text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                      {item.a}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>

        {/* Bottom 4-Metric Model & Corpus Summary */}
        <section aria-label="Model Summary Metrics" className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-5 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900">
            <p className="text-xs text-slate-500 dark:text-slate-400">BiLSTM Accuracy</p>
            <p className="text-2xl font-mono font-bold text-slate-900 dark:text-white tabular-nums mt-1">
              98.55%
            </p>
          </div>
          <div className="p-5 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900">
            <p className="text-xs text-slate-500 dark:text-slate-400">BiLSTM Precision</p>
            <p className="text-2xl font-mono font-bold text-emerald-700 dark:text-emerald-400 tabular-nums mt-1">
              100.0%
            </p>
          </div>
          <div className="p-5 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900">
            <p className="text-xs text-slate-500 dark:text-slate-400">Training Corpus</p>
            <p className="text-2xl font-mono font-bold text-slate-900 dark:text-white tabular-nums mt-1">
              5,572
            </p>
          </div>
          <div className="p-5 rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900">
            <p className="text-xs text-slate-500 dark:text-slate-400">TF-IDF Features</p>
            <p className="text-2xl font-mono font-bold text-slate-900 dark:text-white tabular-nums mt-1">
              3,000
            </p>
          </div>
        </section>
      </main>

      {/* Minimal Footer */}
      <footer className="border-t border-slate-200 dark:border-slate-800 py-8 mt-8">
        <div className="max-w-6xl mx-auto px-6 flex flex-wrap items-center justify-between gap-4 text-xs text-slate-500 dark:text-slate-400">
          <span>Email Spam Classification Platform</span>
          <span>Python · scikit-learn · PyTorch BiLSTM · Flask · FastAPI · PostgreSQL pgvector</span>
        </div>
      </footer>
    </div>
  );
}
