"use client";

import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { Loader2, Save } from "lucide-react";
import { toast } from "sonner";
import { api } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default function AdminSettingsPage() {
  const [heartbeatTimeout, setHeartbeatTimeout] = useState("900");
  const [maxCamerasGlobal, setMaxCamerasGlobal] = useState("50");

  const saveMutation = useMutation({
    mutationFn: (data: Record<string, string>) =>
      api.put("/api/settings/platform", data),
    onSuccess: () => toast.success("Platform settings saved"),
    onError:   () => toast.error("Save failed"),
  });

  function handleSave() {
    saveMutation.mutate({
      heartbeat_timeout_s:   heartbeatTimeout,
      default_max_cameras:   maxCamerasGlobal,
    });
  }

  return (
    <div className="space-y-6 max-w-2xl">
      <div>
        <h1 className="text-xl font-semibold text-zinc-50">Platform Settings</h1>
        <p className="text-sm text-muted-foreground mt-0.5">Global configuration pushed to all nodes</p>
      </div>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Node Health</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">
              Heartbeat timeout (seconds)
            </Label>
            <Input
              type="number"
              value={heartbeatTimeout}
              onChange={(e) => setHeartbeatTimeout(e.target.value)}
              className="bg-zinc-800 border-zinc-700 max-w-xs"
            />
            <p className="text-[10px] text-zinc-600">
              Node is marked OFFLINE after missing {Math.round(parseInt(heartbeatTimeout) / 300)} heartbeat checks
            </p>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Default Limits</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1.5">
            <Label className="text-xs text-muted-foreground">
              Default max cameras per client
            </Label>
            <Input
              type="number"
              value={maxCamerasGlobal}
              onChange={(e) => setMaxCamerasGlobal(e.target.value)}
              className="bg-zinc-800 border-zinc-700 max-w-xs"
            />
          </div>
        </CardContent>
      </Card>

      <Button onClick={handleSave} disabled={saveMutation.isPending} className="flex items-center gap-2">
        {saveMutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
        Save & Broadcast to Nodes
      </Button>
    </div>
  );
}
