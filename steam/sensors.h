/* steam: the sensors. Included once, by steam.c. */
/* ---- sensors ---- */
static char host[64];
static time_t locked_at;
static char temp_path[300], batt_dir[256], fan_path[300];
enum { AUX_BATT, AUX_FAN, AUX_LOADAVG };
static int aux_kind;
static int ncpu = 1;

/* when the screen was locked: swaylock's start time if it's running (so a
 * restart of the saver, e.g. a skip, doesn't reset it), else 0 */
static time_t lock_start(void) {
    time_t best = 0, btime = 0;
    char line[256];
    FILE *f = fopen("/proc/stat", "r");
    if (f) {
        while (fgets(line, sizeof line, f))
            if (sscanf(line, "btime %ld", &btime) == 1) break;
        fclose(f);
    }
    long hz = sysconf(_SC_CLK_TCK);
    DIR *d = btime && hz > 0 ? opendir("/proc") : NULL;
    for (struct dirent *e; d && (e = readdir(d));) {
        if (!isdigit((unsigned char)e->d_name[0])) continue;
        char path[300], comm[64];
        snprintf(path, sizeof path, "/proc/%s/comm", e->d_name);
        slurp_line(path, comm, sizeof comm);
        if (strncmp(comm, "swaylock", 8)) continue;
        snprintf(path, sizeof path, "/proc/%s/stat", e->d_name);
        char st[1024];
        slurp_line(path, st, sizeof st);
        char *p = strrchr(st, ')');
        unsigned long long start = 0;
        /* fields after the command: state is #3, starttime #22 */
        if (p && sscanf(p + 2, "%*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %*s %llu", &start) == 1) {
            time_t t = btime + (time_t)(start / (unsigned long long)hz);
            if (!best || t < best) best = t;
        }
    }
    if (d) closedir(d);
    return best;
}

/* the CPU package temperature: the best-looking sensor on offer */
static void find_temp(void) {
    int best = 0;
    glob_t g;
    if (glob("/sys/class/hwmon/hwmon*/temp*_input", 0, NULL, &g) == 0) {
        for (size_t i = 0; i < g.gl_pathc; i++) {
            const char *in = g.gl_pathv[i];
            char dir[256], name[64], lpath[300], label[64];
            snprintf(dir, sizeof dir, "%.*s", (int)(strrchr(in, '/') - in), in);
            snprintf(lpath, sizeof lpath, "%s/name", dir);
            slurp_line(lpath, name, sizeof name);
            snprintf(lpath, sizeof lpath, "%.*s_label", (int)(strlen(in) - 6), in);
            slurp_line(lpath, label, sizeof label);
            int s = 0;
            if (!strcmp(name, "coretemp") && !strncmp(label, "Package", 7)) s = 100;
            else if (!strcmp(name, "k10temp") && !strcmp(label, "Tdie")) s = 96;
            else if (!strcmp(name, "k10temp") && !strcmp(label, "Tctl")) s = 95;
            else if (!strcmp(name, "zenpower") && !strcmp(label, "Tdie")) s = 94;
            else if (!strcmp(name, "cpu_thermal") || !strcmp(name, "cpu-thermal")) s = 80;
            else if (strstr(label, "CPU") || strstr(label, "Package")) s = 60;
            else if (!strcmp(name, "coretemp") || !strcmp(name, "k10temp")) s = 50;
            else if (!strcmp(name, "acpitz")) s = 20;
            if (s > best) { best = s; snprintf(temp_path, sizeof temp_path, "%s", in); }
        }
        globfree(&g);
    }
    if (glob("/sys/class/thermal/thermal_zone*/type", 0, NULL, &g) == 0) {
        for (size_t i = 0; i < g.gl_pathc; i++) {
            char type[64];
            slurp_line(g.gl_pathv[i], type, sizeof type);
            for (char *c = type; *c; c++) *c = (char)tolower((unsigned char)*c);
            int s = 0;
            if (!strcmp(type, "x86_pkg_temp")) s = 90;
            else if (!strcmp(type, "tcpu")) s = 75;
            else if (strstr(type, "cpu")) s = 70;
            else if (strstr(type, "soc")) s = 40;
            else if (!strcmp(type, "acpitz")) s = 15;
            if (s > best) {
                best = s;
                const char *p = g.gl_pathv[i];
                snprintf(temp_path, sizeof temp_path, "%.*s/temp", (int)(strrchr(p, '/') - p), p);
            }
        }
        globfree(&g);
    }
}

static void find_aux(void) {
    glob_t g;
    if (glob("/sys/class/power_supply/BAT*/capacity", 0, NULL, &g) == 0 && g.gl_pathc) {
        const char *p = g.gl_pathv[0];
        snprintf(batt_dir, sizeof batt_dir, "%.*s", (int)(strrchr(p, '/') - p), p);
        aux_kind = AUX_BATT;
        globfree(&g);
        return;
    }
    if (glob("/sys/class/hwmon/hwmon*/fan*_input", 0, NULL, &g) == 0 && g.gl_pathc) {
        snprintf(fan_path, sizeof fan_path, "%s", g.gl_pathv[0]);
        aux_kind = AUX_FAN;
        globfree(&g);
        return;
    }
    aux_kind = AUX_LOADAVG;
}

static int on_mains(void) {
    glob_t g;
    int on = 0;
    if (glob("/sys/class/power_supply/*/type", 0, NULL, &g) == 0) {
        for (size_t i = 0; i < g.gl_pathc; i++) {
            char t[32], path[300];
            slurp_line(g.gl_pathv[i], t, sizeof t);
            if (strcmp(t, "Mains")) continue;
            snprintf(path, sizeof path, "%.*s/online", (int)(strrchr(g.gl_pathv[i], '/') - g.gl_pathv[i]), g.gl_pathv[i]);
            if (read_long(path, 0) == 1) on = 1;
        }
        globfree(&g);
    }
    return on;
}

/* latest readings */
static struct {
    float temp;                        /* degrees C */
    float load;                        /* 0..1 */
    float mem_gb, mem_total_gb;
    float aux;                         /* battery % / fan rpm / load average */
    int charging, mains;
    double uptime, uptime_at;          /* /proc/uptime, and now() when read */
    unsigned long long cpu_busy, cpu_total;
} sens;

static void read_cpu(unsigned long long *busy, unsigned long long *total) {
    char line[256];
    slurp_line("/proc/stat", line, sizeof line);
    unsigned long long v[8] = {0};
    sscanf(line, "cpu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]);
    *total = 0;
    for (int i = 0; i < 8; i++) *total += v[i];
    *busy = *total - v[3] - v[4];
}

static void read_sensors(void) {
    if (temp_path[0]) sens.temp = read_long(temp_path, 0) / 1000.0f;

    unsigned long long b, t;
    read_cpu(&b, &t);
    if (sens.cpu_total && t > sens.cpu_total)
        sens.load = (float)(b - sens.cpu_busy) / (float)(t - sens.cpu_total);
    sens.cpu_busy = b; sens.cpu_total = t;

    FILE *f = fopen("/proc/meminfo", "r");
    if (f) {
        char line[128];
        long total = 0, avail = 0;
        while (fgets(line, sizeof line, f)) {
            sscanf(line, "MemTotal: %ld", &total);
            sscanf(line, "MemAvailable: %ld", &avail);
        }
        fclose(f);
        sens.mem_total_gb = total / 1048576.0f;
        sens.mem_gb = (total - avail) / 1048576.0f;
    }

    char buf[64], path[300];
    if (aux_kind == AUX_BATT) {
        snprintf(path, sizeof path, "%s/capacity", batt_dir);
        sens.aux = (float)read_long(path, 0);
        snprintf(path, sizeof path, "%s/status", batt_dir);
        slurp_line(path, buf, sizeof buf);
        sens.charging = !strcmp(buf, "Charging");
        sens.mains = on_mains() || sens.charging;
    } else if (aux_kind == AUX_FAN) {
        sens.aux = (float)read_long(fan_path, 0);
    } else {
        slurp_line("/proc/loadavg", buf, sizeof buf);
        sens.aux = strtof(buf, NULL);
    }

    slurp_line("/proc/uptime", buf, sizeof buf);
    sens.uptime = strtod(buf, NULL);
    sens.uptime_at = now();
}


