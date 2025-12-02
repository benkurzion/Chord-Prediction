import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { motion } from "framer-motion";

export default function SongGeneratorUI() {
  const [loading, setLoading] = useState(false);
  const [output, setOutput] = useState<string | null>(null);

  async function handleGenerate() {
    setLoading(true);
    setOutput(null);

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
      });

      const data = await res.json();
      setOutput(data.song || "No output returned");
    } catch (err) {
      setOutput("Error generating song");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col items-center gap-6 p-6 max-w-2xl mx-auto">
      <motion.h1
        className="text-3xl font-bold"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        Song Generator
      </motion.h1>

      <Card className="w-full shadow-lg rounded-2xl">
        <CardContent className="p-6 flex flex-col gap-4">
          <Button
            onClick={handleGenerate}
            disabled={loading}
            className="rounded-xl py-3 text-lg"
          >
            {loading ? "Generating..." : "Generate Song"}
          </Button>

          {output && (
            <motion.pre
              className="whitespace-pre-wrap bg-gray-100 p-4 rounded-xl text-sm"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              {output}
            </motion.pre>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
