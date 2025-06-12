// ------------------------File helper functions------------------------

// returns content of a file as a string
function read_from_file(filename)
{
    local string = null;
    try {
        string = FileToString(filename);
    } catch (e) {
        printl("Error reading from file: " + filename);
    }
    return string;
}

// sends text to file specified in filename
// file gets created in tf/scriptdata folder
function send_to_file(filename, string)
{
    if (typeof string == "string")
        StringToFile(filename, string)
    else
        printl("send_to_file: not a string")
}

// appends text to file specified in filename
function append_to_file(filename,  textToAppend)
{
    if (typeof textToAppend != "string") "append: not a string"

    local file_contents = read_from_file(filename)
    if (file_contents == null)
        file_contents = textToAppend
    else {
        file_contents += (textToAppend)
    }
    send_to_file(filename, file_contents)
}

// sends positions of bots in red team to a string, in which prefix is appended to every line
function shooter_bots_positions_to_string(prefix)
{
    // get all shooter bots
    local bot_list = []
    local ent = null
    while (ent = Entities.FindByClassname(ent, "player")) { //our bots are player class
        // lets assume that bots are in RED team
        if (ent.GetTeam() == TF_TEAM_RED) {
            bot_list.append(ent)
        }
    }
    // get their postitions and put them to string
    local return_string = ""
    foreach(i, ent in bot_list)
    {
        return_string += prefix + " " + ent.entindex() + " " + vector_to_string(ent.GetOrigin()) + "\n"
    }
    // return the string
    return return_string
}