using System.Diagnostics;
using System.Text;
using System.Threading.Tasks;

public static class PythonRunner
{
    public static async Task<(int exitCode, string stdout, string stderr)>
    RunAsync(string pythonExe, string script, string args,
             string workDir = null, bool noWindow = true)
    {
        var psi = new ProcessStartInfo {
            FileName = pythonExe,
            Arguments = $"\"{script}\" {args}",
            WorkingDirectory = string.IsNullOrEmpty(workDir) ? null : workDir,
            CreateNoWindow = noWindow,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError  = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding  = Encoding.UTF8,
        };

        var p = new Process { StartInfo = psi, EnableRaisingEvents = true };

        var sbOut = new StringBuilder();
        var sbErr = new StringBuilder();
        p.OutputDataReceived += (s,e) => { if (e.Data != null) sbOut.AppendLine(e.Data); };
        p.ErrorDataReceived  += (s,e) => { if (e.Data != null) sbErr.AppendLine(e.Data); };

        p.Start();
        p.BeginOutputReadLine();
        p.BeginErrorReadLine();

        await Task.Run(() => p.WaitForExit());
        return (p.ExitCode, sbOut.ToString(), sbErr.ToString());
    }
}
