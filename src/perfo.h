/**********************************************************
*
*			perfo.h
*
*	> Methode de profiling
*
***********************************************************/
#include <stdio.h>
#include <sys/time.h>

#ifndef PERFO_H
#define PERFO_H

typedef struct {
        unsigned int m_start;
        unsigned int m_stop;
        unsigned int m_total;
        char* m_name;
} SPerf;

static unsigned int get_time_in_us()
{
    struct timeval value;
    gettimeofday(&value, NULL);
    return (value.tv_sec*1000000+value.tv_usec);
}
static unsigned int ComputeElapsedTime(SPerf* perfTool)
{ perfTool->m_total += perfTool->m_stop - perfTool->m_start; return perfTool->m_stop - perfTool->m_start;};
static void ResetTimer(SPerf* perfTool){ perfTool->m_start = perfTool->m_stop = 0; };

extern SPerf* CreateSPerf (char* name)
{
    SPerf *perfToolRes = (SPerf*)malloc(sizeof(SPerf));
    perfToolRes->m_start = 0; perfToolRes->m_stop = 0; perfToolRes->m_total = 0; perfToolRes->m_name = name;
    return perfToolRes;
};
extern void DeleteSPerf ( SPerf* perfTool ){ free(perfTool); };
extern void StartProcessing(SPerf* perfTool){ perfTool->m_start = get_time_in_us();};
extern void StopProcessing(SPerf* perfTool) { perfTool->m_stop  = get_time_in_us();};
extern unsigned int GetElapsedTime(SPerf* perfTool)
{
    if ( perfTool->m_start > perfTool->m_stop )
        StopProcessing(perfTool);
    unsigned int res = ComputeElapsedTime(perfTool);
    ResetTimer(perfTool);
    return res;
};
extern unsigned int GetTotalElapsedTime(SPerf* perfTool)
{if ( perfTool->m_start != 0 ) GetElapsedTime(perfTool); return perfTool->m_total;};
extern char* GetName(SPerf* perfTool){ return perfTool->m_name; };



#endif
